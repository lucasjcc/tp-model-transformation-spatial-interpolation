function Y_pred = idw_predict(Train_Dataset, Test_Dataset, params)
% IDW_PREDICT Local or global Inverse-Distance Weighting.
%
% Inputs:
%   Train_Dataset: Struct with finite real .X (n x d) and .Y (n x 1).
%                  Row-vector responses are also accepted.
%   Test_Dataset:  Struct with finite real .X (m x d); .Y is not used.
%   params:        Optional scalar struct with IDW parameters:
%                  - .power: Finite nonnegative exponent (default 2).
%                            Zero gives the mean of the selected neighbors
%                            at queries not coincident with training points.
%                  - .k: Positive integer number of distinct locations.
%                        Omitted or empty uses all locations (global IDW).
%
% Identical coordinate rows are aggregated by their mean response. Distance
% ties are broken by lexicographic coordinate order, independently of input
% row order. At an observed location, its aggregated response is returned.
% If fewer than k distinct locations are available, all are used.
%
% Output:
%   Y_pred:  Vector (m x 1) with estimated values

    if nargin < 3
        params = struct();
    end
    if ~isstruct(params) || ~isscalar(params)
        error('idw_predict:BadParameters', 'IDW parameters must be a scalar struct.');
    end
    if any(~ismember(fieldnames(params), {'power','k'}))
        error('idw_predict:BadParameters', ...
            'Only power and k are configurable.');
    end
    if ~isstruct(Train_Dataset) || ~isscalar(Train_Dataset) || ...
            ~all(isfield(Train_Dataset, {'X', 'Y'})) || ...
            ~isstruct(Test_Dataset) || ~isscalar(Test_Dataset) || ...
            ~isfield(Test_Dataset, 'X')
        error('idw_predict:BadDataset', ...
            'Training data require X and Y; test data require X.');
    end
    X_train = Train_Dataset.X;
    Y_train = Train_Dataset.Y;
    X_test = Test_Dataset.X;
    validateattributes(X_train, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'}, ...
        mfilename, 'Train_Dataset.X');
    validateattributes(Y_train, {'numeric'}, {'vector', 'real', 'finite', ...
        'numel', size(X_train, 1)}, mfilename, 'Train_Dataset.Y');
    validateattributes(X_test, {'numeric'}, {'2d', 'real', 'finite', ...
        'ncols', size(X_train, 2)}, mfilename, 'Test_Dataset.X');
    X_train = double(X_train);
    Y_train = double(Y_train(:));
    X_test = double(X_test);

    if isfield(params, 'power')
        alpha = params.power;
    else
        alpha = 2;
    end
    validateattributes(alpha, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'}, ...
        mfilename, 'params.power');
    alpha = double(alpha);

    % Canonical ordering also makes duplicate aggregation independent of
    % the original row order, including its floating-point summation order.
    [~, order] = sortrows([X_train, Y_train]);
    X_train = X_train(order, :);
    Y_train = Y_train(order);
    [X_train, ~, location_group] = unique(X_train, 'rows', 'sorted');
    Y_train = accumarray(location_group, Y_train, [], ...
        @(values) stable_average(values, ones(size(values)) / numel(values)));
    n_train = size(X_train, 1);
    n_test = size(X_test, 1);
    k = n_train;
    if isfield(params, 'k') && ~isempty(params.k)
        validateattributes(params.k, {'numeric'}, {'scalar', 'real', 'finite', ...
            'integer', 'positive'}, mfilename, 'params.k');
        k = min(double(params.k), n_train);
    end
    Y_pred = zeros(n_test, 1);
    for i = 1:n_test
        x0 = X_test(i, :);
        % Hypot avoids overflow/underflow when squaring coordinates.
        dists = zeros(n_train, 1);
        for dimension = 1:size(X_train, 2)
            dists = hypot(dists, X_train(:, dimension) - x0(dimension));
        end
        if any(~isfinite(dists))
            error('idw_predict:DistanceOverflow', ...
                'Distances exceed the finite numeric range; rescale coordinates.');
        end

        exact_match = dists == 0;
        if any(exact_match)
            values = Y_train(exact_match);
            Y_pred(i) = stable_average(values, ones(size(values)) / numel(values));
            continue;
        end

        % Stable sort preserves the canonical lexicographic order on ties.
        [sorted_dists, idx] = sort(dists);
        current_dists = sorted_dists(1:k);
        current_Y = Y_train(idx(1:k));
        % Equivalent to d^(-alpha), scaled so the nearest weight is one.
        raw_weights = (current_dists(1) ./ current_dists).^alpha;
        lambda = raw_weights / sum(raw_weights);
        Y_pred(i) = stable_average(current_Y, lambda);
    end
end

function value = stable_average(values, weights)
% STABLE_AVERAGE Avoid overflow for finite responses of large magnitude.
%   values and nonnegative, unit-sum weights are matching vectors.
%   value is their scalar weighted mean, limited to the response range.
    scale = max(abs(values));
    if scale == 0
        value = 0;
    else
        normalized_values = values / scale;
        normalized_mean = sum(weights .* normalized_values);
        % Limit only floating-point overshoot beyond the convex hull.
        normalized_mean = min(max(normalized_mean, min(normalized_values)), ...
            max(normalized_values));
        value = scale * normalized_mean;
    end
end
