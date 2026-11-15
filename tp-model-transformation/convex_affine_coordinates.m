function [Q, info] = convex_affine_coordinates(U)
% CONVEX_AFFINE_COORDINATES Orthonormal coordinates of the centered row set.
%   Removes numerical affine dependencies, including constant columns.
%   Based on affine reduction in TP Tool decomp.m.
%   U is a nonempty, finite real matrix whose rows are points.
%   Q contains an orthonormal basis for the centered column space; info
%   reports the affine rank and the numerical rank threshold.
    validateattributes(U, {'numeric'}, {'2d','real','finite','nonempty'});
    U = double(U);
    scale = max(abs(U(:)));
    if scale == 0, scale = 1; end
    centered = U/scale-mean(U/scale,1);
    [basis, values, ~] = svd(centered,'econ');
    values = diag(values);
    threshold = max(size(U))*eps(max(1,norm(U/scale,'fro')));
    affine_rank = sum(values > threshold);
    Q = basis(:,1:affine_rank);
    info.affine_rank = affine_rank;
    info.rank_threshold = threshold;
end
