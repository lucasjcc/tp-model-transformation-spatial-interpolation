function summary = analyze_reviewer3_noise(case_indices, n_repetitions, resume_completed)
% ANALYZE_REVIEWER3_NOISE Evaluate training-response noise with fixed splits.
%   case_indices selects cases 1..10 (default all); n_repetitions is 2..50
%   (default 50).
%   resume_completed (default false) reuses matching validated saved cases.
%   Run from the repository root after the corresponding ablation replay.
%   Uses Random training fraction .3 and noise SD [0,.01,.05,.1,.2] times the
%   clean training SD; test responses stay fixed. Each seed shares one Gaussian
%   vector across levels/methods: TS-G, TS-I, TS-T, Scattered, and control B.
%   summary is a table of per-case/level/method statistics. CSV/MAT records
%   and selected prediction maps are saved under results/reviewer3/cno;
%   reduced runs use an n<repetitions> subdirectory.

    if nargin < 1, case_indices = 1:10; end
    if nargin < 2, n_repetitions = 50; end
    if nargin < 3, resume_completed = false; end
    validateattributes(resume_completed,{'logical'},{'scalar'});
    validateattributes(case_indices, {'numeric'}, {'vector','integer','>=',1,'<=',10});
    validateattributes(n_repetitions, {'numeric'}, {'scalar','integer','>=',2,'<=',50});
    root_dir = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root_dir, 'auxiliary'), fullfile(root_dir, 'interpolation-algorithm'), ...
        fullfile(root_dir, 'tp-model-transformation'));
    output_dir = project_results_path('reviewer3','cno');
    reference_dir = project_results_path('reviewer1','cno','',true);
    if n_repetitions ~= 50
        output_dir = fullfile(output_dir, sprintf('n%d',n_repetitions));
    end
    if ~isfolder(output_dir), mkdir(output_dir); end
    names = ["franke", "peaks", "sinc", "gauss_mixture", repmat("wind_temperature",1,6)];
    months = [1,1,1,1,1:6];
    methods = ["TS-G", "TS-I", "TS-T", "Scattered", "B"];
    fills = ["global_mean", "interp", "tensor_completion"];
    noise_levels = [0, .01, .05, .1, .2];
    ratio = .3;
    seeds = (51:50+n_repetitions)';
    cno_options = struct();
    for case_index = case_indices
        started = tic;
        dataset = get_dataset(20, names(case_index), months(case_index));
        options = spatial_method_options(names(case_index));
        tag = names(case_index);
        if names(case_index) == "wind_temperature"
            tag = sprintf('wind_temperature_month%02d', months(case_index));
        end
        saved_path = fullfile(output_dir,sprintf('noise_%s.mat',tag));
        if resume_completed && isfile(saved_path)
            saved = load(saved_path,'rmse','seeds','noise_levels','methods','metadata','options');
            assert(isequal(size(saved.rmse),[n_repetitions,5,5]) && all(isfinite(saved.rmse),'all'));
            assert(isequal(saved.seeds,seeds) && isequal(saved.noise_levels,noise_levels));
            assert(isequal(saved.methods,methods) && isequal(saved.options,options));
            assert(saved.metadata.zero_noise_max_reproduction_error<1e-12);
            fprintf('%s: reusing complete, validated %d-partition case.\n',tag,n_repetitions);
            continue;
        end
        reference = load(fullfile(reference_dir, ...
            sprintf('ablation_%s_ratio30.mat', tag)), 'rmse');
        rmse = nan(n_repetitions, numel(noise_levels), numel(methods));
        prediction_change = zeros(size(rmse));
        ranks = nan(n_repetitions, numel(noise_levels), 3, 2);
        tensor_change = zeros(n_repetitions, numel(noise_levels), 3);
        noise_sd = nan(n_repetitions, numel(noise_levels));
        noise_realized_rms = nan(size(noise_sd));
        occupied_cells = zeros(n_repetitions,1);
        dimensions = zeros(n_repetitions,2);
        warning_messages = strings(n_repetitions, numel(noise_levels), 3);
        cno_difference = nan(1,3);
        cno_rule_count = nan(1,3);
        for iteration = 1:n_repetitions
            rng(seeds(iteration), 'twister');
            [clean_training, test] = split_dataset(dataset, ratio, names(case_index), 'random');
            algorithm_state = rng;
            scale = std(clean_training.Y, 0);
            assert(isfinite(scale) && scale > 0);
            stream = RandStream('mt19937ar', 'Seed', 100000+seeds(iteration));
            standard_noise = randn(stream, numel(clean_training.Y), 1);
            dims = estimate_grid_size(clean_training.X);
            dimensions(iteration,:) = dims;
            empty_grid = create_grid_from_samples(clean_training, dims, 'none');
            occupied = ~isnan(empty_grid.Y);
            occupied_cells(iteration) = nnz(occupied);
            axes_grid = {unique(empty_grid.X(:,1)), unique(empty_grid.X(:,2))};
            clamped_test = max([axes_grid{1}(1), axes_grid{2}(1)], ...
                min([axes_grid{1}(end), axes_grid{2}(end)], test.X));
            clamped_all = max([axes_grid{1}(1), axes_grid{2}(1)], ...
                min([axes_grid{1}(end), axes_grid{2}(end)], dataset.X));
            baseline_predictions = nan(numel(test.Y), numel(methods));
            baseline_tensors = cell(1,3);
            snapshot = struct();
            save_snapshot = iteration == 1 && ismember(case_index, [1,5]);
            if save_snapshot
                snapshot = struct('dataset', dataset, 'training', clean_training, ...
                    'test', test, 'seed', seeds(iteration), 'ratio', ratio, ...
                    'dims', dims, 'empty_grid', empty_grid, 'options', options, ...
                    'methods', methods, 'predictions', nan(numel(dataset.Y),5), ...
                    'test_rmse', nan(1,5));
                [in_test, test_indices] = ismember(test.X, dataset.X, 'rows');
                assert(all(in_test));
                snapshot.test_indices = test_indices;
            end
            for level_index = 1:numel(noise_levels)
                rng(algorithm_state);
                training = clean_training;
                perturbation = noise_levels(level_index)*scale*standard_noise;
                training.Y = clean_training.Y+perturbation;
                assert(isequal(training.X, clean_training.X));
                noise_sd(iteration,level_index) = noise_levels(level_index)*scale;
                noise_realized_rms(iteration,level_index) = sqrt(mean(perturbation.^2));
                if level_index == 1, assert(isequal(training.Y, clean_training.Y)); end
                for method_index = 1:3
                    lastwarn('');
                    grid_data = create_grid_from_samples(training, dims, fills(method_index));
                    warning_messages(iteration,level_index,method_index) = string(lastwarn);
                    assert(isequal(grid_data.X, empty_grid.X));
                    tensor = reshape(grid_data.Y, dims);
                    assert(all(isfinite(tensor), 'all'));
                    if level_index == 1
                        assert(isequal(grid_data.Y(occupied), empty_grid.Y(occupied)));
                        baseline_tensors{method_index} = tensor;
                    else
                        tensor_change(iteration,level_index,method_index) = ...
                            norm(tensor-baseline_tensors{method_index}, 'fro');
                    end
                    [factors, core] = hosvd_sampled_func(tensor, 0, 1e-2);
                    reconstructed = n_mode_product(n_mode_product(core,factors{1},1),factors{2},2);
                    ranks(iteration,level_index,method_index,:) = [size(factors{1},2),size(factors{2},2)];
                    interpolator = griddedInterpolant(axes_grid, reconstructed, 'linear', 'none');
                    predicted = interpolator(clamped_test(:,1),clamped_test(:,2));
                    rmse(iteration,level_index,method_index) = sqrt(mean((test.Y-predicted).^2));
                    if level_index == 1
                        baseline_predictions(:,method_index) = predicted;
                    else
                        prediction_change(iteration,level_index,method_index) = ...
                            sqrt(mean((predicted-baseline_predictions(:,method_index)).^2));
                    end
                    if save_snapshot && level_index == 1
                        snapshot.predictions(:,method_index) = ...
                            interpolator(clamped_all(:,1),clamped_all(:,2));
                        snapshot.test_rmse(method_index) = rmse(iteration,1,method_index);
                        if method_index == 2
                            snapshot.imputed_grid = grid_data;
                            snapshot.truncated_tensor = reconstructed;
                            snapshot.singular_values = svd(tensor);
                            snapshot.canonical_rank = squeeze(ranks(iteration,1,2,:))';
                        end
                    end
                    if iteration == 1 && level_index == numel(noise_levels)
                        saved_state = rng;
                        [~,~,vertices,weights] = train_ts_model(grid_data,1e-2,1e-6, ...
                            'Convexification','CNO','CNOOptions',cno_options);
                        convex_prediction = estimate_data(test,grid_data,weights,vertices);
                        rng(saved_state);
                        cno_difference(method_index) = max(abs(predicted-convex_prediction.Y));
                        cno_rule_count(method_index) = numel(vertices);
                        assert(cno_difference(method_index) < 1e-6*max(1,max(abs(predicted))));
                    end
                    if method_index == 2
                        before = griddedInterpolant(axes_grid, tensor, 'linear', 'none');
                        predicted_before = before(clamped_test(:,1),clamped_test(:,2));
                        rmse(iteration,level_index,5) = sqrt(mean((test.Y-predicted_before).^2));
                        if level_index == 1
                            baseline_predictions(:,5) = predicted_before;
                        else
                            prediction_change(iteration,level_index,5) = ...
                                sqrt(mean((predicted_before-baseline_predictions(:,5)).^2));
                        end
                        if save_snapshot && level_index == 1
                            snapshot.predictions(:,5) = before(clamped_all(:,1),clamped_all(:,2));
                            snapshot.test_rmse(5) = rmse(iteration,1,5);
                        end
                    end
                end
                direct = interpolation_predict(training,test, ...
                    'Method',options.scattered_method,'Extrapolation','nearest');
                rmse(iteration,level_index,4) = sqrt(mean((test.Y-direct).^2));
                if level_index == 1
                    baseline_predictions(:,4) = direct;
                else
                    prediction_change(iteration,level_index,4) = ...
                        sqrt(mean((direct-baseline_predictions(:,4)).^2));
                end
                if save_snapshot && level_index == 1
                    snapshot.predictions(:,4) = interpolation_predict(training,dataset, ...
                        'Method',options.scattered_method, ...
                        'Extrapolation','nearest');
                    snapshot.test_rmse(4) = rmse(iteration,1,4);
                end
            end
            if save_snapshot
                assert(all(isfinite(snapshot.predictions),'all'));
                from_map = sqrt(mean((snapshot.predictions(snapshot.test_indices,:)-test.Y).^2,1));
                assert(max(abs(from_map-snapshot.test_rmse)) < 1e-12);
                save(fullfile(output_dir,sprintf('snapshot_%s.mat',tag)), '-struct', 'snapshot');
            end
            if mod(iteration,10)==0
                fprintf('%s: %d/%d partitions (%.1fs)\n',tag,iteration,n_repetitions,toc(started));
            end
        end
        assert(all(isfinite(rmse),'all') && all(isfinite(prediction_change),'all'));
        reproduced = squeeze(rmse(:,1,[5,2,4])); % B, TS-I, Scattered.
        zero_noise_error = max(abs(reproduced-reference.rmse(1:n_repetitions,:)),[],'all');
        assert(zero_noise_error < 1e-12);
        metadata = struct('matlab_version', version, 'toolboxes', ver, ...
            'ratio', ratio, ...
            'noise_model', 'Independent Gaussian training-response noise; shared standard vector across levels', ...
            'noise_seed_offset', 100000, 'test_noise_added', false, ...
            'cno_checks', 3, 'zero_noise_max_reproduction_error', zero_noise_error, ...
            'elapsed_seconds', toc(started));
        save(fullfile(output_dir,sprintf('noise_%s.mat',tag)), ...
            'rmse','prediction_change','ranks','tensor_change','noise_sd','noise_realized_rms', ...
            'seeds','noise_levels','methods','dimensions','occupied_cells','options', ...
            'warning_messages','cno_difference','cno_rule_count','metadata');
        fprintf('%s complete: baseline error %.3g, max CNO residual %.3g (%.1fs)\n', ...
            tag,zero_noise_error,max(cno_difference),toc(started));
    end
    % Export only complete, matching datasets; pilots remain separate.
    rows = struct([]);
    for case_index = case_indices
        tag = names(case_index);
        if names(case_index) == "wind_temperature"
            tag = sprintf('wind_temperature_month%02d', months(case_index));
        end
        saved = load(fullfile(output_dir,sprintf('noise_%s.mat',tag)));
        for level_index = 1:numel(noise_levels)
            for method_index = 1:numel(methods)
                values = saved.rmse(:,level_index,method_index);
                deltas = values-saved.rmse(:,1,method_index);
                shift = saved.prediction_change(:,level_index,method_index);
                fallbacks = 0;
                if method_index <= 3
                    fallbacks = sum(contains(saved.warning_messages(:,level_index,method_index), ...
                        'Tensor Completion error'));
                end
                row = struct('dataset',string(tag),'ratio',ratio,'n',n_repetitions, ...
                    'noise_fraction',noise_levels(level_index),'method',methods(method_index), ...
                    'mean_rmse',mean(values),'sd_rmse',std(values), ...
                    'mean_delta',mean(deltas),'sd_delta',std(deltas), ...
                    'relative_mean_delta_pct',100*mean(deltas)/mean(saved.rmse(:,1,method_index)), ...
                    'mean_prediction_shift',mean(shift), ...
                    'mean_noise_sd',mean(saved.noise_sd(:,level_index)), ...
                    'fallbacks',fallbacks);
                rows = [rows;row]; %#ok<AGROW>
            end
        end
    end
    summary = struct2table(rows);
    writetable(summary,fullfile(output_dir,'noise_summary.csv'));
    save(fullfile(output_dir,'noise_summary.mat'),'summary');
end
