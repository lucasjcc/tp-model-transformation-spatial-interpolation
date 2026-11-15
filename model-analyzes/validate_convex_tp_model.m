function report = validate_convex_tp_model(U_rno, S_rno, varargin)
% VALIDATE_CONVEX_TP_MODEL Check nonnegative unit-sum weights and finite vertices.
%   U_rno is a cell array of mode-weight matrices; S_rno is the convex core.
%   Tolerance is the absolute structural tolerance (default 1e-8).
%   report.mode contains minima, row-sum deviations, and per-mode flags;
%   core_tensor_finite and is_valid summarize the checks. This validates
%   node weights, not predictive accuracy; output bounds are checked separately.

    p = inputParser;
    addParameter(p, 'Tolerance', 1e-8, @(x) isscalar(x) && x >= 0);
    parse(p, varargin{:});
    tol = p.Results.Tolerance;

    n_modes = numel(U_rno);
    report.tolerance = tol;
    report.is_valid = true;

    for n = 1:n_modes
        Un = U_rno{n};

        min_value = min(Un(:));
        is_nonneg = min_value >= -tol;

        row_sums = sum(Un, 2);
        max_row_sum_dev = max(abs(row_sums - 1));
        is_partition = max_row_sum_dev <= tol;

        report.mode(n).min_value = min_value;
        report.mode(n).max_row_sum_dev = max_row_sum_dev;
        report.mode(n).is_nonnegative = is_nonneg;
        report.mode(n).is_partition_of_unity = is_partition;

        report.is_valid = report.is_valid && is_nonneg && is_partition;

        fprintf('[validate_convex_tp_model] Mode %d: min(U)=%.3e (%s), max|rowsum-1|=%.3e (%s)\n', ...
            n, min_value, tf_str(is_nonneg), max_row_sum_dev, tf_str(is_partition));
    end

    report.core_tensor_finite = all(isfinite(S_rno(:)));
    report.is_valid = report.is_valid && report.core_tensor_finite;

    if report.is_valid
        fprintf('[validate_convex_tp_model] PASSED: model has finite vertices and convex weights.\n');
    else
        warning('validate_convex_tp_model:invalid', ...
            'Convex TP model assumptions violated -- Propositions bounds/robustness do NOT apply.');
    end
end

function s = tf_str(tf)
% TF_STR Convert scalar logical tf to the diagnostic label 'OK' or 'FAIL'.
%   s is the selected character vector.
    if tf
        s = 'OK';
    else
        s = 'FAIL';
    end
end
