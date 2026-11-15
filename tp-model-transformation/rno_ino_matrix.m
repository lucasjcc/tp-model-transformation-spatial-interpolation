function [Urno, V, info] = rno_ino_matrix(U, tol)
% RNO_INO_MATRIX Convex weights with zero minima and equal column maxima.
%   [W,V,info] satisfies U=W*V. U must be finite and sum-normalized.
%   tol: property/reconstruction tolerance (default 1e-6).
%   Urno contains the resulting weights; V maps them to U. info reports
%   affine rank, reconstruction residual, conditioning, and weight extrema.
%   Constant/dependent inputs are reduced. A constant row set has one unit
%   weight; its INO condition is inapplicable.
%   Adapted from TP Tool rnoino.m/rnodiff.m. See THIRD_PARTY_NOTICES.md.
    if nargin < 2, tol = 1e-6; end
    validateattributes(U, {'numeric'}, {'2d','real','finite','nonempty'});
    validateattributes(tol, {'numeric'}, {'scalar','real','finite','positive'});
    U = double(U);
    if norm(sum(U,2)-1,inf) > max(tol,1e-10)
        error('tp:NotSumNormalized','RNO-INO requires sum-normalized input.');
    end
    [Q, affine_info] = convex_affine_coordinates(U);
    d = size(Q,2);
    if d == 0
        Urno = ones(size(U,1),1);
    else
        if d < size(U,2)-1
            work = sn_nn_matrix(Q,tol);
        else
            work = U;
        end
        r = size(work,2);
        projected = cell(1,r);
        projected{r} = work;
        for k = r:-1:3
            projected{k-1} = project(projected{k}(:,1:k-1),0);
        end
        p = projected{2}(:,1);
        span = max(p)-min(p);
        if span <= 100*eps*max(1,max(abs(p)))
            error('tp:DegenerateProjection','RNO-INO projection lost an affine direction.');
        end
        first = (p-min(p))/span;
        Urno = [first,1-first];
        for k = 3:r
            lifted = Urno-projected{k-1}+projected{k}(:,1:k-1);
            Urno = normalize_step(lifted,tol);
            Urno(:,k) = 1-sum(Urno,2);
        end
    end
    [V, info] = convex_vertex_map(Urno,U,tol);
    info.affine_rank = affine_info.affine_rank;
    info.degenerate_constant = d == 0;
    info.method = 'RNO-INO';
    if d > 0 && (max(abs(info.column_minima)) > max(tol,1e-8) || ...
            max(info.column_maxima)-min(info.column_maxima) > max(10*tol,1e-8))
        error('tp:RNOPropertiesLost','RNO-INO extrema conditions were not achieved.');
    end
end

function A = project(U,a)
% PROJECT Move rows of matrix U toward the unit-sum hyperplane.
%   a is a scalar projection parameter; A has the same size as U.
    A = U+((a-1)/size(U,2))*(sum(U,2)-1);
end

function [W,f] = candidate(U,a)
% CANDIDATE Build normalized weights for matrix U at projection parameter a.
%   W has zero column minima; scalar f measures the missing-column maximum
%   mismatch used by the RNO-INO root search.
    A = project(U,a);
    A = A-min(A,[],1);
    spans = max(A,[],1);
    if any(spans <= 100*eps*max(1,max(abs(A(:)))))
        error('tp:DegenerateProjection','A projected column is constant.');
    end
    W = A./spans;
    W = W/max(sum(W,2));
    f = 1-min(sum(W,2))-max(W(:,1));
end

function W = normalize_step(U,tol)
% NORMALIZE_STEP Find an RNO-INO lifting step by bracketed bisection.
%   U is the lifted factor and tol the root tolerance. W contains normalized
%   weights before appending the final partition-of-unity column.
    left = 0;
    [~, fl] = candidate(U,left);
    if abs(fl) <= min(tol,1e-8)
        [W,~] = candidate(U,left);
        return;
    end
    right = 1;
    [~, fr] = candidate(U,right);
    for expansion = 1:40
        if sign(fl) ~= sign(fr) || abs(fr) <= min(tol,1e-8), break; end
        right = 2*right;
        [~, fr] = candidate(U,right);
    end
    if sign(fl) == sign(fr) && abs(fr) > min(tol,1e-8)
        error('tp:RNOSearchFailed','Could not bracket the RNO-INO root.');
    end
    for iteration = 1:100
        midpoint = (left+right)/2;
        [W,fm] = candidate(U,midpoint);
        if abs(fm) <= min(tol,1e-8), return; end
        if sign(fm) == sign(fl)
            left = midpoint;
            fl = fm;
        else
            right = midpoint;
        end
    end
    error('tp:RNOSearchFailed','RNO-INO normalization did not converge.');
end

