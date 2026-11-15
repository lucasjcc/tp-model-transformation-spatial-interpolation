function [V, info] = convex_vertex_map(W, U, tol)
% CONVEX_VERTEX_MAP Checked least-norm solution of U=W*V (TP Tool contract).
%   Handles rectangular/dependent factors without inverting a forward map.
%   W is a finite convex-weight matrix; U has the same number of sample rows.
%   tol is the relative reconstruction tolerance (default 1e-6). V is the
%   vertex map; info reports residual, rank, conditioning, and weight extrema.
    if nargin < 3, tol = 1e-6; end
    if size(W,1) ~= size(U,1) || isempty(W) || any(~isfinite(W(:)))
        error('tp:InvalidWeights','Weights must be finite with one row per sample.');
    end
    structural_tol = max(1e-10,100*eps*max(size(W)));
    info.min_weight = min(W(:));
    info.row_sum_error = norm(sum(W,2)-1,inf);
    if info.min_weight < -structural_tol || info.row_sum_error > structural_tol
        error('tp:NotConvex','Weights are not nonnegative and sum-normalized.');
    end
    V = lsqminnorm(W,U);
    info.reconstruction_error = norm(W*V-U,'fro')/max(norm(U,'fro'),realmin);
    if any(~isfinite(V(:))) || info.reconstruction_error > max(tol,structural_tol)
        error('tp:ReconstructionLost','Convex factor does not preserve input (relative error %.3g).', ...
            info.reconstruction_error);
    end
    singular_values = svd(W,'econ');
    info.rank = sum(singular_values > max(size(W))*eps(max(singular_values)));
    if info.rank < size(W,2)
        info.condition_number = Inf;
    else
        info.condition_number = singular_values(1)/singular_values(end);
    end
    info.column_minima = min(W,[],1);
    info.column_maxima = max(W,[],1);
end
