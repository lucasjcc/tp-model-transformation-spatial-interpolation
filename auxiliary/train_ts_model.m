function [U_canonical, S_canonical, S_convex, U_convex, Imputed_Tensor, info] = ...
        train_ts_model(Train_Dataset_Grid, singular_value_tolerance, convexification_tolerance, varargin)
% TRAIN_TS_MODEL Fit a scalar 2-D TP model with selectable convexification.
%   [Uc,Sc,S,W,A,info] = train_ts_model(grid,Tsv,Tconv,...)
%   grid.X: complete ndgrid-ordered [N x 2]; grid.Y: [N x 1], NaNs allowed.
%   'Convexification': 'SNNN', 'CNO', 'RNO-INO' (default, case-insensitive).
%   'CNOOptions': options struct for cno_matrix, used only for CNO.
%   singular_value_tolerance is in (0,1]; convexification_tolerance is positive.
%   U_canonical/S_canonical are the truncated factors/core; U_convex/S_convex
%   are the equivalent convex factors/core. Imputed_Tensor is the filled grid.
%   info.type, info.mode and info.reconstruction_error diagnose the hull
%   transformation, separately from HOSVD truncation or imputation error.
%   Each canonical factor satisfies Uc{n}=W{n}*V{n}; V{n} transforms Sc.
    p = inputParser;
    p.PartialMatching = false;
    choices = {'SNNN','CNO','RNO-INO'};
    addParameter(p,'Convexification','RNO-INO', ...
        @(v) ((ischar(v) && isrow(v)) || (isstring(v) && isscalar(v))) && any(strcmpi(v,choices)));
    addParameter(p,'CNOOptions',struct(),@(v) isstruct(v) && isscalar(v));
    parse(p,varargin{:});
    type = upper(char(p.Results.Convexification));
    validateattributes(singular_value_tolerance,{'numeric'}, ...
        {'scalar','real','finite','>',0,'<=',1});
    validateattributes(convexification_tolerance,{'numeric'}, ...
        {'scalar','real','finite','positive'});
    validateattributes(Train_Dataset_Grid.X,{'numeric'},{'2d','real','finite','ncols',2});
    validateattributes(Train_Dataset_Grid.Y,{'numeric'},{'column','real'});
    n_x1 = numel(unique(Train_Dataset_Grid.X(:,1)));
    n_x2 = numel(unique(Train_Dataset_Grid.X(:,2)));
    if n_x1*n_x2 ~= size(Train_Dataset_Grid.Y,1) || ...
            size(Train_Dataset_Grid.X,1) ~= n_x1*n_x2
        error('tp:BadGrid','Grid coordinates and scalar responses must form a complete Cartesian grid.');
    end
    Tensor = reshape(double(Train_Dataset_Grid.Y),[n_x1,n_x2]);
    if any(isinf(Tensor(:)))
        error('tp:BadGrid','Grid responses may contain NaN, but not Inf.');
    end
    if any(isnan(Tensor(:)))
        [U_canonical,S_canonical,Imputed_Tensor] = iHOOI(Tensor,singular_value_tolerance);
    elseif ~any(Tensor(:))
        % Zero tensors have a valid one-component representation. The legacy
        % HOSVD ratio sigma/sigma(1) is undefined when all singular values vanish.
        U_canonical = {ones(n_x1,1)/sqrt(n_x1);ones(n_x2,1)/sqrt(n_x2)};
        S_canonical = 0;
        Imputed_Tensor = Tensor;
    else
        [U_canonical,S_canonical] = hosvd_sampled_func(Tensor,0,singular_value_tolerance);
        Imputed_Tensor = Tensor;
    end
    S_convex = S_canonical;
    U_convex = cell(1,2);
    info.type = type;
    info.mode = cell(1,2);
    for mode = 1:2
        [U_convex{mode},vertices,info.mode{mode}] = convexify_matrix( ...
            U_canonical{mode},type,convexification_tolerance,p.Results.CNOOptions);
        S_convex = n_mode_product(S_convex,vertices,mode);
    end
    original = n_mode_product(n_mode_product(S_canonical,U_canonical{1},1),U_canonical{2},2);
    reconstructed = n_mode_product(n_mode_product(S_convex,U_convex{1},1),U_convex{2},2);
    info.reconstruction_error = norm(reconstructed(:)-original(:))/max(norm(original(:)),realmin);
    if any(~isfinite(S_convex(:))) || info.reconstruction_error > max(1e-10,10*convexification_tolerance)
        error('tp:ReconstructionLost','Convexification did not preserve the canonical tensor.');
    end
end
