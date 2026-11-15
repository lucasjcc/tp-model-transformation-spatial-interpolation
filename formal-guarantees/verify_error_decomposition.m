function report = verify_error_decomposition(f_true_vec, Train_Dataset_Grid, S_rno, U_rno, T_sv, varargin)
% VERIFY_ERROR_DECOMPOSITION Evaluate the three-term grid-node error bound.
%   f_true_vec contains reference values in Train_Dataset_Grid.X row order;
%   the grid also supplies filled Y and dims. S_rno/U_rno are the convex
%   core/factors; T_sv is the HOSVD cutoff. NormType selects a vector norm
%   (default 2, equivalent to the discrete Frobenius norm over grid nodes).
%   report contains rasterization/imputation, HOSVD truncation, and basis
%   reconstruction terms, their total_bound, total_actual_error, and the
%   triangle_inequality_satisfied flag. Reference values may be a proxy field.

    p = inputParser;
    addParameter(p, 'NormType', 2);
    parse(p, varargin{:});
    norm_type = p.Results.NormType;

    dims = Train_Dataset_Grid.dims;
    g_grid = reshape(Train_Dataset_Grid.Y, dims);
    f_true_grid = reshape(f_true_vec, dims);

    % --- Term 1: rasterization + imputation, ||f - g|| ---
    term1 = norm(f_true_grid(:) - g_grid(:), norm_type);

    % --- Term 2: HOSVD truncation, ||g - g_tilde|| ---
    [U, S] = hosvd_sampled_func(g_grid, 0, T_sv);
    g_tilde = S;
    for n = 1:numel(dims)
        g_tilde = n_mode_product(g_tilde, U{n}, n);
    end
    term2 = norm(g_grid(:) - g_tilde(:), norm_type);

    % --- Term 3: basis reconstruction (convex-hull re-parameterization), ||g_tilde - f_hat|| ---
    Query.X = Train_Dataset_Grid.X;
    Query.Y = zeros(size(Query.X, 1), 1);
    Estimated = estimate_data(Query, Train_Dataset_Grid, U_rno, S_rno);
    f_hat_grid_vec = Estimated.Y;
    term3 = norm(g_tilde(:) - f_hat_grid_vec(:), norm_type);

    total_actual = norm(f_true_grid(:) - f_hat_grid_vec(:), norm_type);
    total_bound = term1 + term2 + term3;

    report.term1_rasterization_imputation = term1;
    report.term2_hosvd_truncation = term2;
    report.term3_basis_reconstruction = term3;
    report.total_bound = total_bound;
    report.total_actual_error = total_actual;
    report.triangle_inequality_satisfied = total_actual <= total_bound * (1 + 1e-6) + 1e-9;

    fprintf('[verify_error_decomposition] ||f-g|| (rasterization+imputation) = %.4g\n', term1);
    fprintf('[verify_error_decomposition] ||g-g~|| (HOSVD truncation)        = %.4g\n', term2);
    fprintf('[verify_error_decomposition] ||g~-f^|| (basis reconstruction)   = %.4g %s\n', term3, ...
        ternary(term3 < 1e-8, '(~0, exact re-parameterization confirmed)', ''));
    fprintf('[verify_error_decomposition] sum of terms (bound) = %.4g vs actual ||f-f^|| = %.4g -> %s\n', ...
        total_bound, total_actual, ternary(report.triangle_inequality_satisfied, 'PASSED', 'FAILED'));
end

function out = ternary(cond, a, b)
% TERNARY Return a when scalar cond is true; otherwise return b.
%   a and b are alternative values; out contains the selected value.
    if cond
        out = a;
    else
        out = b;
    end
end
