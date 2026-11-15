function [Train_Dataset, Test_Dataset] = split_dataset(Dataset, train_ratio, dataset_type, sampling_method)
% SPLIT_DATASET Partition X/Y observations into training and test structures.
%   Dataset has [N x 2] X and N responses Y; train_ratio lies in (0,1).
%   sampling_method is random or grid. Grid assumes square meshgrid ordering
%   and rounds the requested training size to a square subgrid; nonsquare
%   sample counts fall back to random. dataset_type is retained but unused.
%   Outputs contain disjoint subsets in their X and Y fields.
    if train_ratio <= 0 || train_ratio >= 1
        error("train_ratio must be between 0 and 1 (e.g., 0.7).");
    end

    n_total_samples = size(Dataset.X, 1);
    indices = 1:n_total_samples;

    if strcmpi(sampling_method, "random")
        rand_indices = randperm(n_total_samples);
        n_train = round(train_ratio * n_total_samples);

        train_idx = rand_indices(1:n_train);
        test_idx  = rand_indices(n_train+1:end);

    elseif strcmpi(sampling_method, "grid")
        side_length = sqrt(n_total_samples);
        
        if floor(side_length) ~= side_length
            warning("Total number of samples (%d) does not form a perfect square. Reverting to random.", n_total_samples);
            [Train_Dataset, Test_Dataset] = split_dataset(Dataset, train_ratio, dataset_type, "random");
            return;
        end
        
        n_train_target = floor(train_ratio * n_total_samples);
        
        train_side_len = round(sqrt(n_train_target));
        
        % Select approximately equally spaced indices, including endpoints.
        idxs_1d = unique(round(linspace(1, side_length, train_side_len)));
        
        train_mask = false(side_length, side_length);
        train_mask(idxs_1d, idxs_1d) = true;
        
        indices_matrix = reshape(indices, side_length, side_length);
        train_idx = indices_matrix(train_mask);
        test_idx = setdiff(indices, train_idx);
        
        fprintf('Grid Sampling: %d x %d sub-grid. Effective ratio = %.2f%% (Target: %.2f%%)\n', ...
             length(idxs_1d), length(idxs_1d), (length(train_idx)/n_total_samples)*100, train_ratio*100);            
    else
        error("Unknown sampling method. Use 'random' or 'grid'.");
    end

    Train_Dataset.X = Dataset.X(train_idx, :);
    Train_Dataset.Y = Dataset.Y(train_idx);

    Test_Dataset.X = Dataset.X(test_idx, :);
    Test_Dataset.Y = Dataset.Y(test_idx);
end
