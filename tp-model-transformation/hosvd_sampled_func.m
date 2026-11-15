function [U,S] = hosvd_sampled_func(P,ord_func,tol)
% HOSVD_SAMPLED_FUNC Compute a mode-wise truncated HOSVD of a dense tensor.
%   P stores sampled values, with ord_func trailing output-space dimensions
%   excluded from decomposition (0 for scalar outputs). tol is the relative
%   singular-value cutoff: retain sigma >= tol*sigma_max in each input mode.
%   U contains the retained orthonormal factors; S is the projected core.
    siz = size(P);
    N = length(siz);
    S = P;
    U = cell(N-ord_func,1);
    for i = 1:(N - ord_func)
        Spd = n_mode(P,i);
        [Ua,Sa] = svd(Spd);
        singular_values = diag(Sa);
        if singular_values(1) == 0
            % Represent a zero tensor with one component per mode. Keep an
            % orthonormal direction instead of comparing undefined 0/0.
            U{i} = Ua(:,1);
        else
            retained = singular_values / singular_values(1) >= tol;
            U{i} = Ua(:,retained);
        end
        S = n_mode_product(S,U{i}',i);
    end
end
