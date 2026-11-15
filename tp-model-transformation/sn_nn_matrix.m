function [Unn, T, info] = sn_nn_matrix(U, tol)
% SN_NN_MATRIX Sum-normalized, nonnegative factor preserving span(U).
%   [W,T,info] = sn_nn_matrix(U,tol)
%   U: finite real [samples x components]. tol: reconstruction tolerance.
%   If info.augmented is false, W=U*T. Otherwise W=[U,1-sum(U,2)]*T.
%   T is a FORWARD map. Use info.vertices for U=W*info.vertices, including
%   augmentation. Based on TP Tool snnn_decomp.m; see THIRD_PARTY_NOTICES.md.
%   Unn contains the convex weights; info also reports reconstruction and
%   weight diagnostics, augmentation status, and the sum-normalization residual.
    if nargin < 2, tol = 1e-6; end
    validateattributes(U, {'numeric'}, {'2d','real','finite','nonempty'});
    validateattributes(tol, {'numeric'}, {'scalar','real','finite','positive'});
    U = double(U);
    [n, r] = size(U);
    coefficients = lsqminnorm(U, ones(n,1));
    sum_error = norm(U*coefficients-1, inf);
    sum_tol = 100*eps*max(n,r);
    augmented = sum_error > sum_tol || max(abs(coefficients)) <= eps;
    if augmented
        Usn = [U, 1-sum(U,2)];
        Tsn = eye(r+1);
    else
        % Replacing one identity column gives determinant coefficients(pivot).
        [~, pivot] = max(abs(coefficients));
        Tsn = eye(r);
        Tsn(:,pivot) = coefficients-1;
        Tsn(pivot,pivot) = Tsn(pivot,pivot)+1;
        Usn = U*Tsn;
    end
    m = size(Usn,2);
    umin = min(0,min(Usn(:)));
    umax = max(2/m,max(Usn(:)));
    c_positive = 1/(1-m*umin);
    c_negative = 1/(1-m*umax);
    if abs(c_negative) > c_positive
        c = c_negative;
    else
        c = c_positive;
    end
    Tnn = ((1-c)/m)*ones(m)+c*eye(m);
    Unn = Usn*Tnn;
    T = Tsn*Tnn;
    [vertices, info] = convex_vertex_map(Unn,U,tol);
    info.vertices = vertices;
    info.augmented = augmented;
    info.sn_residual = sum_error;
    info.method = 'SNNN';
end
