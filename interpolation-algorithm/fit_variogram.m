function var_model = fit_variogram(Train_Dataset, varargin)
% FIT_VARIOGRAM Training-only empirical variogram fit by weighted least squares.
%   'Type': auto (matern32), exponential, matern32, or gaussian.
%   Exponential uses exp(-3*h/range), so its length scale is range/3.
%
% Inputs:
%   Train_Dataset -> struct with .X [N x D] and .Y [N x 1]
%   'NumBins'     -> number of equal-width distance bins for the empirical
%                    variogram (default 15)
%   'MinNuggetFrac' -> lower bound on the nugget as a fraction of the
%                    sample variance of Y for conditioning (default 1e-3)
%
% Output:
%   var_model -> struct with fields .type, .nugget, .sill,
%                .range, plus diagnostic fields .bin_centers,
%                .gamma_empirical, .gamma_fitted, .bin_counts for
%                reporting/plotting the fitted variogram.

    p = inputParser;
    p.PartialMatching = false;
    addParameter(p, 'Type', 'auto', @(s) any(strcmpi(s, {'auto','matern32','exponential','gaussian'})));
    addParameter(p, 'NumBins', 15, @(x) isscalar(x) && x >= 3);
    addParameter(p, 'MinNuggetFrac', 1e-3, @(x) isscalar(x) && x > 0);
    parse(p, varargin{:});
    type = lower(char(p.Results.Type));
    if strcmp(type,'auto')
        type = 'matern32';
    end
    num_bins      = p.Results.NumBins;
    min_nug_frac  = p.Results.MinNuggetFrac;

    X = Train_Dataset.X;
    Y = Train_Dataset.Y(:);
    n = numel(Y);

    var_Y = var(Y);
    if var_Y <= 0 || ~isfinite(var_Y)
        var_Y = eps;
    end
    min_nugget = max(min_nug_frac * var_Y, 1e-12);

    D = spatial_distance(X,X);

    iu = triu(true(n), 1);
    h_pairs = D(iu);
    g_pairs = 0.5 * (Y * ones(1,n) - ones(n,1) * Y') .^ 2;
    g_pairs = g_pairs(iu);

    valid = isfinite(h_pairs) & h_pairs > 0;
    h_pairs = h_pairs(valid);
    g_pairs = g_pairs(valid);

    % Fallback for degenerate/too-small training sets: skip empirical
    % fitting (too few pairs to bin reliably) and use robust heuristics.
    if n < 8 || isempty(h_pairs)
        var_model.type   = type;
        var_model.nugget  = min_nugget;
        var_model.sill    = max(var_Y - min_nugget, min_nugget);
        if isempty(h_pairs)
            var_model.range = 1;
        else
            var_model.range = max(median(h_pairs), eps);
        end
        var_model.bin_centers     = [];
        var_model.gamma_empirical = [];
        var_model.gamma_fitted    = [];
        var_model.bin_counts      = [];
        return;
    end

    % Limit bins to 70% of the maximum lag to reduce sparse far-lag influence.
    max_lag = 0.7 * max(h_pairs);
    edges = linspace(0, max_lag, num_bins + 1);
    bin_centers = (edges(1:end-1) + edges(2:end)) / 2;

    gamma_emp = nan(1, num_bins);
    bin_counts = zeros(1, num_bins);
    for j = 1:num_bins
        if j < num_bins
            in_bin = h_pairs >= edges(j) & h_pairs < edges(j+1);
        else
            in_bin = h_pairs >= edges(j) & h_pairs <= edges(j+1);
        end
        bin_counts(j) = sum(in_bin);
        if bin_counts(j) > 0
            gamma_emp(j) = mean(g_pairs(in_bin));
        end
    end

    keep = bin_counts >= 2 & ~isnan(gamma_emp);
    bin_centers = bin_centers(keep);
    gamma_emp   = gamma_emp(keep);
    bin_counts  = bin_counts(keep);

    if numel(bin_centers) < 3
        % Not enough populated bins for a stable fit: fall back to heuristics.
        var_model.type   = type;
        var_model.nugget  = min_nugget;
        var_model.sill    = max(var_Y - min_nugget, min_nugget);
        var_model.range   = max(median(h_pairs), eps);
        var_model.bin_centers     = bin_centers;
        var_model.gamma_empirical = gamma_emp;
        var_model.gamma_fitted    = model_gamma(bin_centers, var_model.nugget, var_model.sill, var_model.range,type);
        var_model.bin_counts      = bin_counts;
        return;
    end

    % --- Weighted least squares fit (Cressie, 1985/1993 weighting) ---
    % J(theta) = sum_j N_j * [(gamma_emp_j - gamma_model_j(theta)) / gamma_model_j(theta)]^2
    % Parametrized through a log-transform so that sill, range > 0 and
    % nugget >= min_nugget hold for every candidate solution.
    nugget0 = max(0.05 * var_Y, min_nugget);
    sill0   = max(var_Y - nugget0, min_nugget);
    range0  = max(median(h_pairs), eps);

    theta0 = [log(max(nugget0 - min_nugget, eps)); log(sill0); log(range0)];

    objective = @(theta) wls_objective(theta, bin_centers, gamma_emp, bin_counts, min_nugget,type);

    fms_opts = optimset('Display', 'off', 'MaxFunEvals', 2000, 'MaxIter', 2000, 'TolFun', 1e-10, 'TolX', 1e-10);
    theta_opt = fminsearch(objective, theta0, fms_opts);

    [nugget, sill, range] = unpack_theta(theta_opt, min_nugget);

    var_model.type   = type;
    var_model.nugget  = nugget;
    var_model.sill    = sill;
    var_model.range   = range;
    var_model.bin_centers     = bin_centers;
    var_model.gamma_empirical = gamma_emp;
    var_model.gamma_fitted    = model_gamma(bin_centers, nugget, sill, range,type);
    var_model.bin_counts      = bin_counts;
end

function [nugget, sill, range] = unpack_theta(theta, min_nugget)
% UNPACK_THETA Transform three log parameters to valid variogram parameters.
%   theta is a three-element vector; min_nugget is the nugget floor.
%   Outputs are scalar nugget, partial sill, and range.
    nugget = min_nugget + exp(theta(1));
    sill   = exp(theta(2));
    range  = exp(theta(3));
end

function J = wls_objective(theta, h, gamma_emp, N, min_nugget,type)
% WLS_OBJECTIVE Evaluate the count-weighted relative semivariogram residual.
%   theta contains log parameters; h, gamma_emp, and N are matching bin
%   vectors; min_nugget and type define the model. J is a finite scalar cost.
    [nugget, sill, range] = unpack_theta(theta, min_nugget);
    gamma_model = model_gamma(h, nugget, sill, range,type);
    gamma_model = max(gamma_model, 1e-12);
    resid = (gamma_emp - gamma_model) ./ gamma_model;
    J = sum(N(:) .* resid(:).^2);
    if ~isfinite(J), J = 1e100; end
end

function gamma = model_gamma(h, nugget, sill, range,type)
% MODEL_GAMMA Evaluate a Matérn-3/2, exponential, or Gaussian semivariogram.
%   h contains lags; nugget, sill, and range are scalar parameters; type
%   selects the kernel. gamma matches h in size and is zero at zero lag.
    h = abs(h);
    switch type
        case 'matern32'
            z = sqrt(3)*h/range;
            contribution = -expm1(log1p(z)-z);
            small = z < 1e-4;
            t = z(small);
            contribution(small) = t.^2.*(0.5-t/3+t.^2/8-t.^3/30);
            gamma = nugget+sill*contribution;
        case 'exponential'
            gamma = nugget-sill*expm1(-3*h/range);
        case 'gaussian'
            gamma = nugget-sill*expm1(-3*(h/range).^2);
    end
    gamma(h == 0) = 0;
end
