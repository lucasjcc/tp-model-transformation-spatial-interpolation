function y_pred = interpolation_predict(Train_Dataset, Test_Dataset, varargin)
% INTERPOLATION_PREDICT Delaunay-based scattered-data interpolation.
%
% Inputs:
%   Train_Dataset: finite real .X [n x d], d=2 or 3, and .Y [n x 1].
%                  Row-vector responses are also accepted. Identical
%                  coordinates are aggregated by their mean response.
%   Test_Dataset:  finite real .X [m x d]; .Y is not used.
% Name-value options:
%   'Method': 'natural' (default), 'linear', or 'nearest'. Natural-neighbor
%             interpolation is not linear barycentric interpolation.
%   'Extrapolation': 'nearest' (default), 'linear', 'none', or 'boundary'
%                    (boundary requires MATLAB R2024a or later).
% Output:
%   y_pred: [m x 1]; 'none' returns NaN outside the convex hull.
%
% At least d+1 distinct, affinely independent locations are required.
% Canonical ordering makes construction independent of input row order for
% a fixed MATLAB version.

    p = inputParser;
    p.PartialMatching = false;
    addParameter(p, 'Method', 'natural', ...
        @(value) is_text_choice(value, {'natural', 'linear', 'nearest'}));
    addParameter(p, 'Extrapolation', 'nearest', ...
        @(value) is_text_choice(value, {'nearest', 'linear', 'none', 'boundary'}));
    parse(p, varargin{:});
    opts = p.Results;

    if ~isstruct(Train_Dataset) || ~isscalar(Train_Dataset) || ...
            ~all(isfield(Train_Dataset, {'X', 'Y'})) || ...
            ~isstruct(Test_Dataset) || ~isscalar(Test_Dataset) || ...
            ~isfield(Test_Dataset, 'X')
        error('interpolation_predict:BadDataset', ...
            'Training data require X and Y; test data require X.');
    end
    X_train = Train_Dataset.X;
    Y_train = Train_Dataset.Y;
    X_test = Test_Dataset.X;
    validateattributes(X_train, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'}, ...
        mfilename, 'Train_Dataset.X');
    dimension = size(X_train, 2);
    if ~ismember(dimension, [2, 3])
        error('interpolation_predict:BadDimension', 'Coordinates must have two or three columns.');
    end
    validateattributes(Y_train, {'numeric'}, {'vector', 'real', 'finite', ...
        'numel', size(X_train, 1)}, mfilename, 'Train_Dataset.Y');
    validateattributes(X_test, {'numeric'}, {'2d', 'real', 'finite', ...
        'ncols', dimension}, mfilename, 'Test_Dataset.X');
    X_train = double(X_train);
    Y_train = double(Y_train(:));
    X_test = double(X_test);

    [~, order] = sortrows([X_train, Y_train]);
    X_train = X_train(order, :);
    Y_train = Y_train(order);
    [X_train, ~, location_group] = unique(X_train, 'rows', 'sorted');
    Y_train = accumarray(location_group, Y_train, [], @stable_mean);
    if size(X_train, 1) < dimension + 1
        error('interpolation_predict:InsufficientPoints', ...
            'At least d+1 distinct training locations are required.');
    end

    % Scale only the rank diagnostic, without changing interpolation geometry.
    coordinate_scale = max(abs(X_train), [], 1);
    coordinate_scale(coordinate_scale == 0) = 1;
    diagnostic_coordinates = X_train ./ coordinate_scale;
    if rank(diagnostic_coordinates - diagnostic_coordinates(1, :)) < dimension
        error('interpolation_predict:DegenerateGeometry', ...
            'Training locations are collinear in 2-D or coplanar in 3-D.');
    end
    if isempty(X_test)
        y_pred = zeros(0, 1);
        return;
    end

    F = scatteredInterpolant(X_train, Y_train, ...
        lower(char(opts.Method)), lower(char(opts.Extrapolation)));

    y_pred = F(X_test);
    y_pred = y_pred(:);
    if any(isinf(y_pred)) || (any(isnan(y_pred)) && ~strcmpi(opts.Extrapolation, 'none'))
        error('interpolation_predict:InvalidPrediction', ...
            'Interpolation produced nonfinite predictions; inspect coordinate scaling and geometry.');
    end
end

function valid = is_text_choice(value, choices)
% IS_TEXT_CHOICE Check whether scalar text value matches an entry in choices.
%   choices is a cell array of names; valid is a scalar logical.
    valid = ((ischar(value) && isrow(value)) || ...
        (isstring(value) && isscalar(value))) && any(strcmpi(value, choices));
end

function value = stable_mean(values)
% STABLE_MEAN Avoid overflow when averaging duplicate finite responses.
%   values is a nonempty numeric vector; value is its scalar mean.
    scale = max(abs(values));
    if scale == 0
        value = 0;
    else
        value = scale * mean(values / scale);
    end
end
