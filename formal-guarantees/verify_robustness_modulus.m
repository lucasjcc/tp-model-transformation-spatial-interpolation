function report = verify_robustness_modulus(S_rno, U_rno, Train_Dataset_Grid, Query_Dataset, varargin)
% VERIFY_ROBUSTNESS_MODULUS Test sensitivity to convex-core perturbations.
%   S_rno/U_rno are the scalar core and fixed convex factors.
%   Train_Dataset_Grid.X defines the grid box; Query_Dataset.X is [N x 2].
%   Magnitudes (default [1e-3,1e-2,1e-1]) scales uniform perturbations by the
%   core range (unit scale for a constant core). NumTrials defaults to 20 per
%   magnitude; Seed defaults to 42. Only in-domain queries are evaluated.
%   report.trials records bounds max(abs(Delta)), observed deviations,
%   satisfaction flags, and tightness ratios; all_passed and
%   mean_tightness_ratio summarize the trials. No model retraining is tested.

    p = inputParser;
    addParameter(p, 'Magnitudes', [1e-3, 1e-2, 1e-1]);
    addParameter(p, 'NumTrials', 20, @(x) isscalar(x) && x >= 1);
    addParameter(p, 'Seed', 42);
    parse(p, varargin{:});
    magnitudes = p.Results.Magnitudes;
    n_trials = p.Results.NumTrials;

    rng_state = rng();
    rng(p.Results.Seed);

    grid_x1 = unique(Train_Dataset_Grid.X(:,1));
    grid_x2 = unique(Train_Dataset_Grid.X(:,2));
    X = Query_Dataset.X;
    in_domain = X(:,1) >= min(grid_x1) & X(:,1) <= max(grid_x1) & ...
                X(:,2) >= min(grid_x2) & X(:,2) <= max(grid_x2);
    Query_InDomain.X = X(in_domain, :);
    Query_InDomain.Y = zeros(size(Query_InDomain.X, 1), 1);

    if isempty(Query_InDomain.X)
        error('verify_robustness_modulus:noInDomainPoints', ...
            'No query points fall inside the training grid domain Omega.');
    end

    Baseline = estimate_data(Query_InDomain, Train_Dataset_Grid, U_rno, S_rno);
    f_baseline = Baseline.Y;

    S_range = range(S_rno(:));
    if S_range == 0
        S_range = 1; % degenerate constant core tensor; avoid a zero-scale perturbation
    end

    idx = 0;
    n_conditions = numel(magnitudes) * n_trials;
    trials(n_conditions) = struct('magnitude', 0, 'bound', 0, ...
        'observed_max_deviation', 0, 'satisfied', false, 'tightness_ratio', 0);

    for m = 1:numel(magnitudes)
        mag = magnitudes(m);
        for t = 1:n_trials
            idx = idx + 1;
            Delta = (2*rand(size(S_rno)) - 1) * mag * S_range;

            S_perturbed = S_rno + Delta;
            Perturbed = estimate_data(Query_InDomain, Train_Dataset_Grid, U_rno, S_perturbed);

            observed = max(abs(Perturbed.Y - f_baseline));
            bound = max(abs(Delta(:)));

            trials(idx).magnitude = mag;
            trials(idx).bound = bound;
            trials(idx).observed_max_deviation = observed;
            trials(idx).satisfied = observed <= bound * (1 + 1e-6) + 1e-9;
            trials(idx).tightness_ratio = observed / max(bound, eps);
        end
    end

    rng(rng_state);

    report.trials = trials;
    report.all_passed = all([trials.satisfied]);
    report.mean_tightness_ratio = mean([trials.tightness_ratio]);

    fprintf('[verify_robustness_modulus] %d trials across %d magnitude levels\n', n_conditions, numel(magnitudes));
    if report.all_passed
        fprintf('[verify_robustness_modulus] PASSED: bound |f''-f| <= max|Delta B| held in all trials (mean tightness = %.3f)\n', ...
            report.mean_tightness_ratio);
    else
        n_fail = sum(~[trials.satisfied]);
        fprintf('[verify_robustness_modulus] FAILED: %d/%d trials violated the modulus bound.\n', n_fail, n_conditions);
    end
end
