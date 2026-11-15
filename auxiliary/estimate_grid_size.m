function [grid_size, info] = estimate_grid_size(X, varargin)
    % ESTIMATE_GRID_SIZE Robust grid node counts from full-space neighbors.
    %
    % Usage:
    %   [grid_size, info] = estimate_grid_size(X, 'Alpha', 1.5)
    %
    % Inputs:
    %   X -> finite, real [N x D] training coordinates, N >= 1, D >= 1.
    %   'Alpha' -> finite oversampling factor > 1 (default 1.5).
    %   'NeighborCount' -> positive integer (default 3). Use the median
    %       distance to this distinct neighbor; fewer are used if necessary.
    %       This reduces sensitivity to isolated close pairs, but does not
    %       guarantee robustness to clusters larger than NeighborCount.
    %
    % Outputs:
    %   grid_size -> [1 x D] node counts, including both domain endpoints.
    %       Constant axes (within roundoff) have one node.
    %   info -> requested/effective spacing, unclipped node counts, limits,
    %       clipping flags, unique sample count and neighbor diagnostics.
    % Exact duplicate rows do not affect spacing or the size limit. Repeated
    % coordinates in one projection do not imply zero full-space distance.
    % The neighbor statistic and limits are heuristics, not accuracy bounds.

    validateattributes(X, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'});
    X = double(X);
    p = inputParser;
    p.PartialMatching = false;
    addParameter(p, 'Alpha', 1.5, @(v) isnumeric(v) && isreal(v) && ...
        isscalar(v) && isfinite(v) && v > 1);
    addParameter(p, 'NeighborCount', 3, @(v) isnumeric(v) && isreal(v) && ...
        isscalar(v) && isfinite(v) && v >= 1 && v == fix(v));
    parse(p, varargin{:});
    alpha = p.Results.Alpha;
    [n_samples, n_dims] = size(X);

    [unique_X, ~, sample_index] = unique(X, 'rows');
    n_unique = size(unique_X, 1);
    range_k = max(X, [], 1) - min(X, [], 1);
    if any(~isfinite(range_k))
        error('grid:CoordinateRangeOverflow', 'Coordinate ranges must be finite.');
    end
    coordinate_tolerance = 32 * eps(max(abs(X), [], 1));
    constant_axes = range_k <= coordinate_tolerance;

    axis_lengths = range_k;
    distance_tolerance = norm(coordinate_tolerance(~constant_axes));
    axis_lengths(constant_axes) = 0;
    if any(~isfinite(axis_lengths))
        error('grid:CoordinateRangeOverflow', 'Axis lengths must be finite.');
    end

    nearest_distances = zeros(n_unique, 1);
    neighbor_distances = zeros(n_unique, 1);
    neighbor_count_used = zeros(n_unique, 1);
    if n_unique > 1 && any(~constant_axes)
        distances = squareform(pdist(unique_X));
        if any(~isfinite(distances(:)))
            error('grid:DistanceOverflow', 'Neighbor distances must be finite.');
        end
        % Exclude self-distances and separations indistinguishable from zero.
        distances(distances <= distance_tolerance) = Inf;
        distances(1:n_unique+1:end) = Inf;
        distances = sort(distances, 2);
        available = sum(isfinite(distances), 2);
        neighbor_count_used = min(p.Results.NeighborCount, available);
        valid = available > 0;
        nearest_distances(valid) = distances(valid, 1);
        rows = find(valid);
        neighbor_distances(valid) = distances(sub2ind(size(distances), ...
            rows, neighbor_count_used(valid)));
    end

    usable = neighbor_distances > 0;
    spacing = 0;
    if any(usable)
        spacing = median(neighbor_distances(usable));
    end
    requested_step = spacing / alpha;
    if spacing > 0 && requested_step == 0
        error('grid:SpacingUnderflow', 'Alpha is too large for the coordinate scale.');
    end

    % Counts refer to nodes: I nodes span I-1 intervals. Limits use unique
    % observations and active dimensions, so duplicating rows cannot refine
    % the grid. The minimum takes priority for very small training sets.
    L_min = 5;
    n_active = max(1, nnz(~constant_axes));
    L_max = max(L_min, ceil(2 * n_unique^(1 / n_active)));
    I_est_k = ones(1, n_dims);
    if requested_step > 0
        I_est_k(~constant_axes) = ceil(min( ...
            axis_lengths(~constant_axes) / requested_step, realmax - 1)) + 1;
    else
        % No separable neighbors: use the minimum on nonconstant axes.
        I_est_k(~constant_axes) = L_min;
    end
    grid_size = max(L_min, min(I_est_k, L_max));
    grid_size(constant_axes) = 1;
    effective_step = zeros(1, n_dims);
    effective_step(~constant_axes) = axis_lengths(~constant_axes) ./ ...
        (grid_size(~constant_axes) - 1);

    info.d_i = nearest_distances(sample_index);
    info.d_bar_k = repmat(spacing, 1, n_dims); % legacy field: now full-space median
    info.Delta_t_k = repmat(requested_step, 1, n_dims);
    info.Delta_t_k(constant_axes) = 0;
    info.I_est_k = I_est_k;
    info.L_min = L_min;
    info.L_max = L_max;
    info.n_samples = n_samples;
    info.n_unique_samples = n_unique;
    info.constant_axes = constant_axes;
    info.coordinate_tolerance = coordinate_tolerance;
    info.axis_lengths = axis_lengths;
    info.spacing = spacing;
    info.neighbor_count = p.Results.NeighborCount;
    info.neighbor_count_used = neighbor_count_used;
    info.neighbor_distances = neighbor_distances;
    info.insufficient_neighbors = ~any(usable);
    info.lower_limit_active = ~constant_axes & I_est_k < L_min;
    info.upper_limit_active = ~constant_axes & I_est_k > L_max;
    info.effective_step_k = effective_step;
    info.total_nodes = prod(grid_size);
end
