function report = verify_certified_bounds(S_rno, U_rno, Train_Dataset_Grid, Query_Dataset, varargin)
% VERIFY_CERTIFIED_BOUNDS Check queried outputs against convex core extrema.
%   S_rno is the scalar-output core; U_rno contains two convex weight matrices.
%   Train_Dataset_Grid.X defines the grid box; Query_Dataset.X is [N x 2].
%   Tolerance is the allowed absolute overshoot (default 1e-9).
%   report contains extrema, query counts, violations, maximum overshoot,
%   and passed. External queries are counted but excluded; an empty in-domain
%   set returns passed=true without testing predictions. These checks concern
%   model bounds, not field accuracy outside or inside the sampled domain.

    p = inputParser;
    addParameter(p, 'Tolerance', 1e-9, @(x) isscalar(x) && x >= 0);
    parse(p, varargin{:});
    tol = p.Results.Tolerance;

    b_min = min(S_rno(:));
    b_max = max(S_rno(:));

    grid_x1 = unique(Train_Dataset_Grid.X(:,1));
    grid_x2 = unique(Train_Dataset_Grid.X(:,2));

    X = Query_Dataset.X;
    in_domain = X(:,1) >= min(grid_x1) & X(:,1) <= max(grid_x1) & ...
                X(:,2) >= min(grid_x2) & X(:,2) <= max(grid_x2);

    Query_InDomain.X = X(in_domain, :);
    Query_InDomain.Y = zeros(size(Query_InDomain.X, 1), 1);

    report.b_min = b_min;
    report.b_max = b_max;
    report.n_query_total  = size(X, 1);
    report.n_in_domain    = sum(in_domain);
    report.n_out_of_domain = sum(~in_domain);

    if report.n_in_domain == 0
        warning('verify_certified_bounds:noInDomainPoints', ...
            'No query points fall inside the training grid domain Omega; nothing to check.');
        report.n_violations = 0;
        report.violation_fraction = NaN;
        report.max_violation_magnitude = 0;
        report.passed = true;
        return;
    end

    Estimated = estimate_data(Query_InDomain, Train_Dataset_Grid, U_rno, S_rno);
    y_hat = Estimated.Y;

    below = y_hat < (b_min - tol);
    above = y_hat > (b_max + tol);
    violations = below | above;

    deficits = zeros(size(y_hat));
    deficits(below) = b_min - y_hat(below);
    deficits(above) = y_hat(above) - b_max;

    report.n_violations = sum(violations);
    report.violation_fraction = report.n_violations / report.n_in_domain;
    report.max_violation_magnitude = max(deficits);
    report.passed = report.n_violations == 0;

    fprintf('[verify_certified_bounds] Omega-bounds = [%.6g, %.6g]\n', b_min, b_max);
    fprintf('[verify_certified_bounds] %d/%d in-domain query points checked (%d excluded as outside Omega)\n', ...
        report.n_in_domain, report.n_query_total, report.n_out_of_domain);
    if report.passed
        fprintf('[verify_certified_bounds] PASSED: 0 violations.\n');
    else
        fprintf('[verify_certified_bounds] FAILED: %d/%d violations (%.4f%%), max overshoot = %.3e\n', ...
            report.n_violations, report.n_in_domain, 100*report.violation_fraction, report.max_violation_magnitude);
    end
end
