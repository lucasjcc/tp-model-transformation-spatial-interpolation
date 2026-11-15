function D = spatial_distance(A,B)
% SPATIAL_DISTANCE Compute pairwise distances between sample locations.
%   A is [N x d], B is [M x d], and D is the [N x M] distance matrix.
    validateattributes(A,{'numeric'},{'2d','real','finite'});
    validateattributes(B,{'numeric'},{'2d','real','finite','ncols',size(A,2)});
    D = pdist2(double(A),double(B));
end
