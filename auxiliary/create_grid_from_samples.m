function [Train_Dataset_Grid, grid_stats] = create_grid_from_samples(Train_Dataset, n_bins_vec, fill_method, varargin)
    % CREATE_GRID_FROM_SAMPLES Aggregate samples, then impute empty grid cells.
    % Inputs:
    %   Train_Dataset: X contains [N x 2] coordinates; Y contains N responses.
    %   n_bins_vec: [nx, ny] grid node counts.
    %   fill_method: interp (default), linear, natural, global_mean, kriging,
    %       knn, tensor_completion, or none (retain missing values).
    %   'KrigingModel': struct forwarded to kriging_predict for that fill.
    %   'NeighborCount': number of nearest neighbors for knn (default 5).
    %   'PreserveSampleGrid': false (default). True keeps the original axes
    %       of a complete Cartesian sample grid, including nonuniform axes.
    %       n_bins_vec must match its distinct node counts. No imputation is
    %       needed; repeated observations at a node are averaged.
    % Outputs:
    %   Train_Dataset_Grid: grid coordinates, responses, centers and dims.
    %       For virtual grids (PreserveSampleGrid=false), constant axes
    %       within roundoff collapse to one node, even with explicit
    %       multi-node requests. Interpolation on the remaining axis is
    %       linear, with nearest-boundary extrapolation.
    %   grid_stats: optional diagnostics AFTER within-cell averaging and
    %               BEFORE imputation: .dims, .total_cells, .empty_cells,
    %               .empty_cells_pct (0 to 100). Empty means no observations.
    
    X_train = Train_Dataset.X;
    Y_train = Train_Dataset.Y;

    if nargin < 3
        fill_method = 'interp';
    end
    % Preserve the former numeric KNN argument and manual Kriging struct.
    if isscalar(varargin) && isnumeric(varargin{1}) && strcmpi(fill_method,'knn')
        varargin = {'NeighborCount',varargin{1}};
    elseif isscalar(varargin) && isstruct(varargin{1}) && strcmpi(fill_method,'kriging')
        varargin = {'KrigingModel',varargin{1}};
    end
    p = inputParser;
    p.PartialMatching = false;
    addParameter(p,'KrigingModel',struct(),@(v) isstruct(v) && isscalar(v));
    addParameter(p,'NeighborCount',5,@(v) isnumeric(v) && isscalar(v) && ...
        isfinite(v) && v >= 1 && v == fix(v));
    addParameter(p,'PreserveSampleGrid',false,@(v) ...
        (islogical(v) || isnumeric(v)) && isscalar(v) && isreal(v) && ...
        isfinite(v) && (v == 0 || v == 1));
    parse(p,varargin{:});
    options = p.Results;

    % Relative tolerance used by low-rank tensor completion.
    singular_value_tolerance = 1e-2;

    n_bins_vec = reshape(n_bins_vec, 1, []); 
    if options.PreserveSampleGrid
        validateattributes(X_train,{'numeric'},{'2d','real','finite','nonempty','ncols',2});
        validateattributes(n_bins_vec,{'numeric'},{'vector','numel',2,'integer','positive','finite'});
        validateattributes(Y_train,{'numeric'},{'vector','real','finite','numel',size(X_train,1)});
        Y_train = Y_train(:);
    end
    
    centers = cell(1, 2);
    min_v = min(X_train, [], 1);
    max_v = max(X_train, [], 1);
    % Virtual grids collapse constant axes to avoid repeated centers and
    % invalid histogram edges. Preserved grids use exact node membership.
    constant_axes = (max_v - min_v) <= 32 * eps(max(abs(X_train), [], 1));
    if ~options.PreserveSampleGrid
        n_bins_vec(constant_axes) = 1;
    end
    
    for i = 1:2
        if options.PreserveSampleGrid
            centers{i} = unique(X_train(:,i))';
            if numel(centers{i}) ~= n_bins_vec(i)
                error('grid:SampleGridMismatch', ...
                    'Preserved sample axes must match the requested node counts.');
            end
        elseif constant_axes(i)
            centers{i} = min_v(i);
        else
            centers{i} = linspace(min_v(i), max_v(i), n_bins_vec(i));
        end
    end
    if options.PreserveSampleGrid && ...
            size(unique(X_train,'rows'),1) ~= prod(n_bins_vec)
        error('grid:IncompleteSampleGrid', ...
            'PreserveSampleGrid requires a complete Cartesian sample grid.');
    end
    
    % Tensor modes follow ndgrid ordering: mode 1 is x1, mode 2 is x2.
    [X1_grid, X2_grid] = ndgrid(centers{1}, centers{2});
    
    dims = n_bins_vec; 

    subs = zeros(size(X_train));
    
    for i = 1:2
        if options.PreserveSampleGrid
            % Exact node membership also preserves adjacent floating-point
            % coordinates whose midpoint would round onto an endpoint.
            [~,~,loc] = unique(X_train(:,i));
        else
            edges = [-inf, (centers{i}(1:end-1) + centers{i}(2:end))/2, inf];
            [~, ~, loc] = histcounts(X_train(:, i), edges);
        end
        subs(:, i) = loc;
    end
    
    % Keep bin indices within the dimensions accepted by accumarray.
    subs(:,1) = min(subs(:,1), dims(1));
    subs(:,2) = min(subs(:,2), dims(2));
    subs(subs < 1) = 1;
    
    grid_sum    = accumarray(subs, Y_train, dims, @sum, NaN);
    grid_count  = accumarray(subs, 1, dims, @sum, 0);
    
    Y_grid = grid_sum ./ grid_count; 

    % Preserve occupancy before any fill method can conceal empty cells.
    grid_stats = struct('dims', dims, 'total_cells', prod(dims), ...
        'empty_cells', nnz(grid_count == 0), ...
        'empty_cells_pct', 100 * nnz(grid_count == 0) / prod(dims));
    
    mask_nan = isnan(Y_grid);
    mask_filled = ~mask_nan;
    
    if ~any(mask_nan(:))
        Y_final = Y_grid;
    else
        grid_coordinates = [X1_grid(:), X2_grid(:)];
        X_filled = grid_coordinates(mask_filled(:), :);
        Y_filled = Y_grid(mask_filled);
        Y_filled = Y_filled(:);
        X_target = grid_coordinates(mask_nan(:), :);
        
        if isempty(X_filled)
             warning('Empty grid. Returning global mean.');
             Y_final = repmat(mean(Y_train), dims);
        else
            Data_Known = struct('X', X_filled, 'Y', Y_filled);
            Data_Query = struct('X', X_target);
            switch lower(fill_method)
                case 'none'
                    % Retain NaNs for completion during model training.
                    Y_predicted = NaN(size(X_target, 1), 1);

                case {'interp', 'linear', 'natural'}
                    interpolation_method = 'natural';
                    if strcmpi(fill_method, 'linear')
                        interpolation_method = 'linear';
                    end
                    Y_predicted = interpolate_grid_points(Data_Known, Data_Query, ...
                        interpolation_method, constant_axes);
                    
                case 'kriging'
                    try
                        Y_predicted = kriging_predict(Data_Known, Data_Query, options.KrigingModel);
                    catch
                        Y_predicted = interpolate_grid_points(Data_Known, Data_Query, ...
                            'linear', constant_axes);
                    end

                case 'global_mean'
                    % Weight original observations, not occupied cells, equally.
                    mean_val = mean(Y_train, 'omitnan');
                    Y_predicted = repmat(mean_val, size(X_target, 1), 1);

                case 'knn'
                    k_neighbors = options.NeighborCount;
                    
                    k_neighbors = min(k_neighbors, size(X_filled, 1));
                    
                    distances = spatial_distance(X_target,X_filled);
                    [~,idx] = sort(distances,2);
                    idx = idx(:,1:k_neighbors);
                    
                    neighbor_vals = Y_filled(idx);
                    
                    Y_predicted = mean(neighbor_vals, 2);
                    
                case 'tensor_completion'
                    % Complete the grid through adaptive low-rank HOOI.
                    try
                        [~, ~, Y_completed_grid, ~] = iHOOI(Y_grid, singular_value_tolerance);

                        % Preserve observed cell means.
                        Y_predicted = Y_completed_grid(mask_nan);
                    catch ME
                        warning('Tensor Completion error (iHOOI): %s. Reverting to interpolation.', ME.message);
                        Y_predicted = interpolate_grid_points(Data_Known, Data_Query, ...
                            'natural', constant_axes);
                    end
                    
                otherwise
                    error('Unknown method.');
            end
            
            Y_final = Y_grid;
            Y_final(mask_nan) = Y_predicted;
        end
    end

    Train_Dataset_Grid.Y = reshape(Y_final, [], 1);
    Train_Dataset_Grid.X = [X1_grid(:), X2_grid(:)];
    Train_Dataset_Grid.centers = centers;
    Train_Dataset_Grid.dims = dims;
end

function predictions = interpolate_grid_points(known, query, method, constant_axes)
    % INTERPOLATE_GRID_POINTS Fill query nodes, including collapsed grids.
    %   known has X/Y samples; query.X contains targets. method selects
    %   interpolation; constant_axes marks collapsed coordinate axes.
    %   predictions is one response per query, with boundary clamping in 1-D.
    % A collapsed grid cannot form a 2-D triangulation. Interpolate along
    % its active axis, or use the only known value if one cell is occupied.
    active = find(~constant_axes);
    if isempty(active) || (isscalar(active) && isscalar(known.Y))
        predictions = repmat(known.Y(1), size(query.X, 1), 1);
    elseif isscalar(active)
        coordinates = known.X(:, active);
        targets = max(min(coordinates), min(max(coordinates), query.X(:, active)));
        predictions = interp1(coordinates, known.Y, targets, 'linear');
    else
        predictions = interpolation_predict(known, query, ...
            'Method', method, 'Extrapolation', 'nearest');
    end
end
