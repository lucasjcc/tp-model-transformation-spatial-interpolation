function Estimated_Dataset = estimate_data(Test_Dataset, Train_Dataset_Grid, weights, S, varargin)
% ESTIMATE_DATA Evaluate a scalar 2-D convex TP model using linear weights.
%   Test_Dataset.X: finite [N x 2]; Y is not needed.
%   Train_Dataset_Grid.X: original grid coordinates; weights={W1,W2}.
%   S: scalar-output core of size [size(W1,2),size(W2,2)].
%   'Extrapolation': 'clamp' (default) limits queries to the grid box and
%   extends its boundary values; 'error' rejects external queries.
%   Output X retains ORIGINAL queries; Y is [N x 1]; outside_grid records
%   which queries lie outside the box. Clamping does not certify field
%   accuracy outside the sampled domain.
    p = inputParser;
    p.PartialMatching = false;
    addParameter(p,'Extrapolation','clamp', ...
        @(v) ((ischar(v) && isrow(v)) || (isstring(v) && isscalar(v))) && any(strcmpi(v,{'clamp','error'})));
    parse(p,varargin{:});
    X = Test_Dataset.X;
    validateattributes(X,{'numeric'},{'2d','real','finite','ncols',2});
    validateattributes(S,{'numeric'},{'2d','real','finite','nonempty'});
    if ~iscell(weights) || numel(weights) ~= 2
        error('tp:BadWeights','Scalar 2-D prediction requires two weight matrices.');
    end
    axes = {unique(Train_Dataset_Grid.X(:,1)),unique(Train_Dataset_Grid.X(:,2))};
    for mode = 1:2
        validateattributes(weights{mode},{'numeric'},{'2d','real','finite','nonempty'});
        if size(weights{mode},1) ~= numel(axes{mode}) || size(weights{mode},2) ~= size(S,mode)
            error('tp:BadWeights','Factor dimensions do not match grid/core in mode %d.',mode);
        end
        if min(weights{mode}(:)) < -1e-10 || norm(sum(weights{mode},2)-1,inf) > 1e-8
            error('tp:NotConvex','Prediction requires convex weights at the grid nodes.');
        end
    end
    lower = [axes{1}(1),axes{2}(1)];
    upper = [axes{1}(end),axes{2}(end)];
    outside = any(X < lower | X > upper,2);
    if strcmpi(p.Results.Extrapolation,'error') && any(outside)
        error('tp:OutsideGrid','Queries outside the grid box are not allowed.');
    end
    queries = max(lower,min(upper,double(X)));
    Estimated_Dataset.X = X;
    Estimated_Dataset.outside_grid = outside;
    Estimated_Dataset.Y = zeros(size(X,1),1);
    for i = 1:size(X,1)
        mu = cell(1,2);
        for mode = 1:2
            if isscalar(axes{mode})
                mu{mode} = weights{mode}(1,:);
            else
                mu{mode} = interp1(axes{mode},weights{mode},queries(i,mode),'linear');
            end
        end
        Estimated_Dataset.Y(i) = mu{1}*S*mu{2}';
    end
end
