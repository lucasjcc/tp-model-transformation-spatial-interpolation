function report = hosvd_truncation_bound(Grid_Tensor, T_sv, ord_func)
% HOSVD_TRUNCATION_BOUND Compute and check the discrete HOSVD error bound.
%   Grid_Tensor is a dense imputed tensor; T_sv is the relative singular-value
%   cutoff. ord_func counts trailing output-space dimensions (default 0).
%   The Frobenius error is bounded by the root-sum-square of discarded
%   singular values across decomposed modes (de Lathauwer et al., 2000).
%   report contains T_sv, retained ranks, singular values, bound, actual_error,
%   satisfied, and tightness_ratio (actual/bound, with an epsilon denominator).

    if nargin < 3
        ord_func = 0;
    end

    siz = size(Grid_Tensor);
    N = ndims(Grid_Tensor) - ord_func;

    discarded_sq_sum = 0;
    ranks = zeros(1, N);
    singular_values = cell(1, N);

    for n = 1:N
        Xn = n_mode(Grid_Tensor, n);
        s = svd(Xn);
        smax = s(1);
        if smax == 0
            ranks(n) = 1;
            singular_values{n} = s;
            continue;
        end
        retained = s >= T_sv * smax;
        r_n = sum(retained);
        if r_n < 1
            r_n = 1;
            retained(1) = true;
        end
        discarded = s(~retained);
        discarded_sq_sum = discarded_sq_sum + sum(discarded.^2);
        ranks(n) = r_n;
        singular_values{n} = s;
    end

    bound = sqrt(discarded_sq_sum);

    [U, S] = hosvd_sampled_func(Grid_Tensor, ord_func, T_sv);
    Recon = S;
    for n = 1:N
        Recon = n_mode_product(Recon, U{n}, n);
    end

    actual_error = norm(Grid_Tensor(:) - Recon(:));

    report.T_sv = T_sv;
    report.ranks = ranks;
    report.singular_values = singular_values;
    report.bound = bound;
    report.actual_error = actual_error;
    report.satisfied = actual_error <= bound * (1 + 1e-6) + 1e-9;
    report.tightness_ratio = actual_error / max(bound, eps);

    fprintf('[hosvd_truncation_bound] T_sv=%.3g, retained ranks = %s\n', T_sv, mat2str(ranks));
    if report.satisfied
        fprintf('[hosvd_truncation_bound] PASSED: ||g-g~||_F = %.4g <= bound = %.4g (tightness = %.3f)\n', ...
            actual_error, bound, report.tightness_ratio);
    else
        fprintf('[hosvd_truncation_bound] FAILED: ||g-g~||_F = %.4g > bound = %.4g\n', actual_error, bound);
    end
end
