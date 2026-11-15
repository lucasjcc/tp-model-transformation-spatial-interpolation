function [Y_pred, Var_est, var_model] = kriging_predict(Train_Dataset, Test_Dataset, var_model)
% KRIGING_PREDICT Estimate scalar responses using ordinary Kriging.
% Usage:
%   [Y_pred, Var_est, var_model] = kriging_predict(Train_Dataset, Test_Dataset)
%       -> Gaussian semivariogram, nugget=0, range=1.5, and
%          sill=var(Train_Dataset.Y).
%   [Y_pred, Var_est] = kriging_predict(Train_Dataset, Test_Dataset, var_model)
%       -> optionally overrides defaults with a complete model struct
%          (.nugget, .sill, .range, and optional .type).
%          .fit_variogram: true requests training-only WLS fitting of the
%          chosen type. Complete manual parameters bypass fitting by default.
%          Exponential covariance is sill*exp(-3*h/range), length scale range/3.
%
% Inputs:
%   Train_Dataset: struct with .X [n x d] and .Y [n x 1].
%   Test_Dataset:  struct with .X [m x d]; test responses are not used.
%   var_model:    optional scalar struct of semivariogram parameters.
% Outputs:
%   Y_pred:    [m x 1] estimated values.
%   Var_est:   [m x 1] ordinary Kriging estimation variance.
%   var_model: model actually used, including fitting diagnostics when fitted.

    X_train = Train_Dataset.X;
    Y_train = Train_Dataset.Y(:);
    X_test = Test_Dataset.X;
    n_train = size(X_train, 1);
    n_test = size(X_test, 1);

    if nargin < 3 || isempty(var_model)
        var_model = struct();
    end
    if ~isstruct(var_model) || ~isscalar(var_model)
        error('kriging_predict:BadCall', ...
            'Pass variogram parameters as a scalar struct.');
    end
    validateattributes(X_train, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'});
    validateattributes(Y_train, {'numeric'}, {'vector', 'real', 'finite', 'numel', n_train});
    validateattributes(X_test, {'numeric'}, {'2d', 'real', 'finite', 'ncols', size(X_train, 2)});

    allowed_fields = {'type','nugget','sill','range','fit_variogram', ...
        'bin_centers','gamma_empirical','gamma_fitted','bin_counts'};
    if any(~ismember(fieldnames(var_model),allowed_fields))
        error('kriging_predict:BadParameters','Unknown semivariogram parameter.');
    end
    if ~isfield(var_model,'type') || isempty(var_model.type)
        var_model.type = 'gaussian';
    end
    fit_requested = false;
    if isfield(var_model,'fit_variogram')
        validateattributes(var_model.fit_variogram,{'logical','numeric'},{'scalar','binary'});
        fit_requested = logical(var_model.fit_variogram);
    end
    dists_train = spatial_distance(X_train,X_train);
    dists_test = spatial_distance(X_train,X_test);
    if fit_requested && var(Y_train) > 0
        var_model = fit_variogram(Train_Dataset,'Type',var_model.type);
        var_model.fit_variogram = true;
    else
        if ~isfield(var_model,'nugget'), var_model.nugget = 0; end
        if ~isfield(var_model,'sill'), var_model.sill = var(Y_train); end
        if ~isfield(var_model,'range')
            var_model.range = 1.5;
        end
        var_model.fit_variogram = false;
    end
    validateattributes(var_model.nugget, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'});
    validateattributes(var_model.sill, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'});
    validateattributes(var_model.range, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});

    % A constant response gives a zero semivariogram under the benchmark.
    % Predict that constant without adding artificial variance or a nugget.
    model_scale = var_model.nugget + var_model.sill;
    if model_scale == 0
        if any(Y_train ~= Y_train(1))
            error('kriging_predict:ZeroVariogram', ...
                'A zero semivariogram requires constant training responses.');
        end
        Y_pred = repmat(Y_train(1), n_test, 1);
        Var_est = zeros(n_test, 1);
        return;
    end

    K_gamma = calculate_variogram(dists_train, var_model) / model_scale;
    Gamma_0 = calculate_variogram(dists_test, var_model) / model_scale;

    % Solve Gamma*w + mu*1 = Gamma_0 subject to 1'*w = 1. Eliminate
    % the constraint before the pseudoinverse to preserve unbiasedness when
    % the zero-nugget Gaussian system is numerically rank deficient.
    % Common scaling leaves weights unchanged; no nugget is added.
    uniform_weights = ones(n_train, 1) / n_train;
    basis = null(ones(1, n_train));
    reduced_gamma = basis' * K_gamma * basis;
    reduced_rhs = basis' * (Gamma_0 - K_gamma * uniform_weights);
    reduced_gamma = (reduced_gamma + reduced_gamma') / 2;
    [directions, eigenvalues] = eig(reduced_gamma, 'vector');
    tolerance = max(size(reduced_gamma)) * eps(max(abs(eigenvalues), [], 'all'));
    keep = abs(eigenvalues) > tolerance;
    % Apply the symmetric pseudoinverse through its eigenvectors rather
    % than forming a dense inverse, which magnifies cancellation errors.
    reduced_solution = directions(:, keep) * ...
        ((directions(:, keep)' * reduced_rhs) ./ eigenvalues(keep));
    w = uniform_weights + basis * reduced_solution;
    w = w + (1 - sum(w, 1)) / n_train;

    Y_pred = w' * Y_train;
    % Equivalent to mu + w'*Gamma_0, using the reduced system to avoid
    % cancellation between large weights in nearly singular systems.
    base_variance = 2 * mean(Gamma_0, 1) ...
        - uniform_weights' * K_gamma * uniform_weights;
    Var_est = model_scale * ...
        (base_variance + sum(reduced_solution .* reduced_rhs, 1))';
end

function gamma = calculate_variogram(h, params)
% CALCULATE_VARIOGRAM Evaluate the semivariogram; gamma(0)=0.
%   h contains lags; params has type, nugget, partial sill, and positive range.
%   gamma has the same size as h.
    h = abs(h);
    nugget = params.nugget;
    sill = params.sill; % partial sill (contribution)
    range = params.range;

    switch lower(params.type)
        case 'matern32'
            z = sqrt(3) * h / range;
            contribution = -expm1(log1p(z)-z);
            small = z < 1e-4;
            t = z(small);
            contribution(small) = t.^2.*(0.5-t/3+t.^2/8-t.^3/30);
            gamma = nugget+sill*contribution;
        case 'spherical'
            idx = h <= range;
            gamma = zeros(size(h));
            gamma(idx) = nugget + sill * (1.5 * (h(idx) / range) - 0.5 * (h(idx) / range).^3);
            gamma(~idx) = nugget + sill;
        case 'exponential'
            gamma = nugget - sill * expm1(-3 * h / range);
        case 'gaussian'
            gamma = nugget - sill * expm1(-3 * (h / range).^2);
        otherwise
            error('kriging_predict:UnknownVariogram', 'Unknown variogram type.');
    end
    % A nugget is a discontinuity as h approaches zero, not gamma(0).
    gamma(h == 0) = 0;
end
