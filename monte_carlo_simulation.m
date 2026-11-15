%% Monte Carlo interpolation benchmark
% Run from the repository root. Compares three TS imputation variants with
% Kriging, IDW, Scattered interpolation, and a neural network across selected
% datasets/months, training fractions, and sampling methods. Uses the same
% model defaults as main.m. Reports error metrics,
% rule counts, grid occupancy, and numerical checks of convex-model bounds.
% No plots are created. CSV/MAT exports overwrite matching files in results_dir;
% use a separate directory for reduced validation runs.
addpath('./tp-model-transformation/');
addpath('./auxiliary/');
addpath('./interpolation-algorithm/');
addpath('./datasets/');
addpath('./model-analyzes/');
addpath('./formal-guarantees/');

clearvars; close all; clc;

%% Experiment settings
% train_ratios_list = [0.1, 0.3, 0.5];
train_ratios_list = 0.1;

% Option: ["random", "grid"]. Random sampling changes the partition per repetition. "grid" is deterministic.
sampling_methods_list = "random";
n_mc_iterations = 3;

n_samples_per_input = 20;

avaliable_datasets = [
    "franke"
    "peaks"
    "sinc"
    "gauss_mixture"
    "wind_temperature"
];
% Choose any subset: e.g. avaliable_datasets([1, 2, 5]), or all with
% avaliable_datasets(1:end). Names may also be given as a string array.
chosen_datasets = avaliable_datasets([1, 2, 3, 4, 5]);

% NOAA months to simulate: 1=January, ..., 12=December.
% Example: chosen_months = [1, 2, 3, 6]. Ignored for synthetic datasets.
chosen_months = [1, 2, 3, 4, 5, 6];

chosen_datasets = reshape(string(chosen_datasets), 1, []);
if isempty(chosen_datasets) || any(~ismember(chosen_datasets, avaliable_datasets))
    error('mc:BadDatasets', 'Choose at least one dataset from avaliable_datasets.');
end
chosen_datasets = unique(chosen_datasets, 'stable');
if any(chosen_datasets == "wind_temperature")
    validateattributes(chosen_months, {'numeric'}, {'vector', 'real', 'finite', ...
        'integer', '>=', 1, '<=', 12, 'nonempty'}, mfilename, 'chosen_months');
    chosen_months = unique(reshape(double(chosen_months), 1, []), 'stable');
end

% Expand the selections into one case per synthetic dataset or NOAA month.
% Canonical month tags prevent CSV and MAT files from overwriting each other.
simulation_cases = struct('dataset', {}, 'month', {}, 'tag', {});
for dataset_idx = 1:numel(chosen_datasets)
    selected_name = char(chosen_datasets(dataset_idx));
    if strcmp(selected_name, 'wind_temperature')
        months_to_run = chosen_months;
    else
        months_to_run = NaN; % month does not apply to synthetic datasets
    end
    for month_idx = 1:numel(months_to_run)
        selected_month = months_to_run(month_idx);
        if isnan(selected_month)
            selected_tag = selected_name;
        else
            selected_tag = sprintf('%s_month%02d', selected_name, selected_month);
        end
        simulation_cases(end+1) = struct('dataset', selected_name, ...
            'month', selected_month, 'tag', selected_tag); %#ok<SAGROW>
    end
end

% TS model filling (imputation) variants:
%   global_mean       -> TS-G
%   interp            -> TS-I
%   tensor_completion -> TS-T
ts_fill_variants   = ["global_mean", "interp", "tensor_completion"];
ts_variant_names   = ["TS (Global Mean)", "TS (Interp)", "TS (Tensor Comp)"];

% TS model hyperparameters
singular_value_tolerance   = 1e-2;   % T_sv
convexification_tolerance  = 1e-6;
convexification_type = "CNO"; % "SNNN", "CNO", or "RNO-INO"
cno_options = struct(); % Used only for CNO.
convexification_type = string(validatestring(convexification_type, {'SNNN','CNO','RNO-INO'}));

% Benchmark parameters
idw_power = 2;
idw_k     = []; % Global IDW: use all distinct training locations.
nn_hidden_neurons = 20;
nn_train_fcn      = 'trainbr';

% Verification of formal guarantees for convex TS models
verify_guarantees = true;

% Exporting numerical results
export_results = true;
% Separate sampling schemes and repetition counts to protect reference batches.
run_tag = sprintf('%s_n%d', char(strjoin(reshape(sampling_methods_list,1,[]),'_')), n_mc_iterations);
results_dir = project_results_path('monte_carlo', convexification_type, run_tag);
if export_results && ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Accumulators use this fixed method order; TS variants start at index 5.
model_names_final = [
    "Kriging"
    "IDW"
    "Scattered"
    "Neural Net"
    "TS (Global Mean)"
    "TS (Interp)"
    "TS (Tensor Comp)"
];
n_methods = numel(model_names_final);

% Metrics collected (same as main.m). Rules = model complexity.
metric_names = ["RE", "MAE", "RMSE", "StdErr", "MaxAE", "Rules"];
n_metrics = numel(metric_names);
IDX_TS_FIRST = 5;   % first model index corresponding to a TS variant

% Grid diagnostics are separate from prediction errors: one row per
% replication, measured after sample aggregation and before imputation.
grid_stat_names = ["GridX1", "GridX2", "GridCells", "EmptyCellsPct"];

% Names of verified formal guarantees (convex TS models only)
guarantee_names = [
    "Validity"
    "CertifiedBounds"
    "RobustnessModulus"
    "HOSVDTruncation"
    "ErrorDecomposition"
];
n_guarantees = numel(guarantee_names);

n_ratios = numel(train_ratios_list);

fprintf('Starting Monte Carlo Simulation: %d dataset/month cases, %d iterations per configuration.\n', ...
    numel(simulation_cases), n_mc_iterations);

%% Dataset and month cases
for case_idx = 1:numel(simulation_cases)
    dataset_name = simulation_cases(case_idx).dataset;
    current_month = simulation_cases(case_idx).month;
    dataset_tag = simulation_cases(case_idx).tag;

    spatial_options = spatial_method_options(dataset_name);
    idw_params = struct('power', idw_power, 'k', idw_k);
    kriging_options = struct('type',spatial_options.kriging_type, ...
        'fit_variogram',spatial_options.kriging_fit);
    grid_options = {'KrigingModel',kriging_options};
    Dataset = get_dataset(n_samples_per_input, dataset_name, current_month);

    % Reset raw results for this case: each MAT contains only its dataset/month.
    all_results = struct('dataset', {}, 'month', {}, 'train_ratio', {}, 'sampling', {}, ...
        'convexification', {}, 'convexification_diagnostics', {}, ...
        'spatial_options', {}, 'kriging_models', {}, ...
        'accumulated_errors', {}, 'grid_statistics', {}, ...
        'guarantee_pass', {}, 'guarantee_quant', {});

    fprintf('\nCase %d/%d: %s | Formal guarantees: %s\n', ...
        case_idx, numel(simulation_cases), dataset_tag, string(verify_guarantees));
    disp(spatial_options);

    for tr_idx = 1:length(train_ratios_list)
        current_ratio = train_ratios_list(tr_idx);

        for sm_idx = 1:length(sampling_methods_list)
            current_sampling = sampling_methods_list(sm_idx);

            fprintf('\n------------------------------------------------------------\n');
            fprintf('Simulating: Ratio = %.2f | Sampling = %s\n', current_ratio, current_sampling);
            fprintf('------------------------------------------------------------\n');

            if strcmpi(current_sampling, "grid") && n_mc_iterations > 1
                warning('mc:deterministicSampling', ...
                    ['"Grid" sampling is deterministic: all %d replications use the same ' ...
                     'train/test partition (std~0 except for stochastic methods). ' ...
                     'Consider n_mc_iterations = 1 for this mode.'], n_mc_iterations);
            end

            % Failed fits remain NaN and are excluded from aggregate metrics.
            accumulated_errors = nan(n_methods, n_metrics, n_mc_iterations);
            grid_statistics = nan(n_mc_iterations, numel(grid_stat_names));
            convexification_diagnostics = cell(numel(ts_fill_variants), n_mc_iterations);
            kriging_models = cell(1,n_mc_iterations);
            guarantee_pass     = nan(numel(ts_fill_variants), n_guarantees, n_mc_iterations);
            guarantee_quant    = nan(numel(ts_fill_variants), 5, n_mc_iterations);

            %% Repeated fits
            for k = 1:n_mc_iterations
                seed = k + n_mc_iterations * ((tr_idx - 1) + n_ratios * (sm_idx - 1));
                rng(seed);

                [Train_Dataset, Test_Dataset] = split_dataset(Dataset, current_ratio, dataset_name, current_sampling);

                % Kriging.
                try
                    [y_pred_krig, ~, kriging_models{k}] = kriging_predict( ...
                        Train_Dataset, Test_Dataset,kriging_options);
                    accumulated_errors(1, :, k) = compute_metrics(Test_Dataset.Y, y_pred_krig, NaN);
                catch ME
                    warning('mc:krigingFailed', 'Kriging failed (rep %d): %s', k, ME.message);
                end

                % Global IDW.
                try
                    y_pred_idw = idw_predict(Train_Dataset, Test_Dataset, idw_params);
                    accumulated_errors(2, :, k) = compute_metrics(Test_Dataset.Y, y_pred_idw, NaN);
                catch ME
                    warning('mc:idwFailed', 'IDW failed (rep %d): %s', k, ME.message);
                end

                % Scattered interpolation with the dataset-specific method.
                try
                    y_pred_scat = interpolation_predict(Train_Dataset, Test_Dataset, ...
                        'Method',spatial_options.scattered_method,'Extrapolation','nearest');
                    accumulated_errors(3, :, k) = compute_metrics(Test_Dataset.Y, y_pred_scat, NaN);
                catch ME
                    warning('mc:scatteredFailed', 'Scattered failed (rep %d): %s', k, ME.message);
                end

                % Neural network.
                try
                    y_pred_nn = neural_network_predict(Train_Dataset, Test_Dataset, ...
                        'HiddenNeurons', nn_hidden_neurons, 'TrainFcn', nn_train_fcn, ...
                        'ShowWindow', false);
                    accumulated_errors(4, :, k) = compute_metrics(Test_Dataset.Y, y_pred_nn, NaN);
                catch ME
                    warning('mc:nnFailed', 'Neural Net failed (rep %d): %s', k, ME.message);
                end

                % TS variants share a training grid and differ in imputation.
                if strcmpi(current_sampling, "grid")
                    grid_size = [length(unique(Train_Dataset.X(:,1))) length(unique(Train_Dataset.X(:,2)))];
                else
                    grid_size = estimate_grid_size(Train_Dataset.X);
                end
                % A non-square reference can trigger split_dataset's Random
                % fallback; only complete sample grids preserve their axes.
                preserve_sample_grid = strcmpi(current_sampling,'grid') && ...
                    size(unique(Train_Dataset.X,'rows'),1) == prod(grid_size);

                % The TS variants share the same rasterized grid. Measure it
                % without imputation so even failed fills/fits cannot bias
                % occupancy statistics toward successful TS predictions.
                try
                    [~, grid_stats] = create_grid_from_samples(Train_Dataset, grid_size, 'none', ...
                        'PreserveSampleGrid',preserve_sample_grid);
                    grid_statistics(k, :) = [grid_stats.dims, grid_stats.total_cells, ...
                        grid_stats.empty_cells_pct];
                catch ME
                    warning('mc:gridStatisticsFailed', ...
                        'Grid diagnostics failed (rep %d): %s', k, ME.message);
                end

                for ts_idx = 1:length(ts_fill_variants)
                    variant = ts_fill_variants(ts_idx);
                    model_row = IDX_TS_FIRST + ts_idx - 1;

                    try
                        Train_Dataset_Grid = create_grid_from_samples( ...
                            Train_Dataset,grid_size,char(variant),grid_options{:}, ...
                            'PreserveSampleGrid',preserve_sample_grid);

                        [~, ~, S_rno, U_rno, Imputed_Tensor, convexification_info] = train_ts_model(Train_Dataset_Grid, ...
                            singular_value_tolerance, convexification_tolerance, ...
                            'Convexification', convexification_type, 'CNOOptions', cno_options);
                        convexification_diagnostics{ts_idx,k} = convexification_info;

                        Train_Dataset_Grid.Y = Imputed_Tensor(:);

                        Estimated_Dataset = estimate_data(Test_Dataset, Train_Dataset_Grid, U_rno, S_rno);

                        number_rules = prod(size(S_rno)); %#ok<PSIZE>

                        accumulated_errors(model_row, :, k) = ...
                            compute_metrics(Test_Dataset.Y, Estimated_Dataset.Y, number_rules);

                        % Verify structure and bounds without changing the model.
                        if verify_guarantees
                            try
                                [~, guarantee_report] = evalc(['run_formal_guarantee_checks(' ...
                                    'Dataset, Train_Dataset_Grid, Test_Dataset, S_rno, U_rno, ' ...
                                    'singular_value_tolerance)']);

                                guarantee_pass(ts_idx, 1, k) = double(guarantee_report.model_validity.is_valid);
                                guarantee_pass(ts_idx, 2, k) = double(guarantee_report.certified_bounds.passed);
                                guarantee_pass(ts_idx, 3, k) = double(guarantee_report.robustness_modulus.all_passed);
                                guarantee_pass(ts_idx, 4, k) = double(guarantee_report.hosvd_truncation.satisfied);
                                guarantee_pass(ts_idx, 5, k) = double(guarantee_report.error_decomposition.triangle_inequality_satisfied);

                                guarantee_quant(ts_idx, 1, k) = guarantee_report.robustness_modulus.mean_tightness_ratio;
                                guarantee_quant(ts_idx, 2, k) = guarantee_report.hosvd_truncation.tightness_ratio;
                                guarantee_quant(ts_idx, 3, k) = guarantee_report.error_decomposition.term1_rasterization_imputation;
                                guarantee_quant(ts_idx, 4, k) = guarantee_report.error_decomposition.term2_hosvd_truncation;
                                guarantee_quant(ts_idx, 5, k) = guarantee_report.error_decomposition.term3_basis_reconstruction;
                            catch ME
                                warning('mc:guaranteeFailed', 'Formal guarantees failed (%s, rep %d): %s', variant, k, ME.message);
                            end
                        end
                    catch ME
                        warning('mc:tsFailed', 'TS (%s) failed (rep %d): %s', variant, k, ME.message);
                    end
                end

                if mod(k, 5) == 0
                    fprintf('.');
                end
            end
            fprintf('\n');

            %% Aggregate repetition statistics
            mean_errors = mean(accumulated_errors, 3, 'omitnan');
            std_errors  = std(accumulated_errors, 0, 3, 'omitnan');

            n_ok = squeeze(sum(~isnan(accumulated_errors(:, 3, :)), 3));
            if any(n_ok < n_mc_iterations)
                warning('mc:incompleteReplications', ...
                    'Not all methods completed the %d replications; refer to N_ok column.', ...
                    n_mc_iterations);
            end

            metric_cols = cell(1, n_metrics);
            for m = 1:n_metrics
                metric_cols{m} = format_mean_std(mean_errors(:, m), std_errors(:, m));
            end
            Results_Table = table(metric_cols{:}, n_ok, ...
                'RowNames', cellstr(model_names_final), ...
                'VariableNames', [reshape(cellstr(metric_names), 1, []), {'N_ok'}]);

            % Arithmetic means over constructed grids, rather than the
            % product of mean dimensions or a pooled empty-cell percentage.
            % Numeric columns preserve small nonzero percentages. NaN marks
            % benchmark methods that do not use the TS grid.
            valid_grids = all(isfinite(grid_statistics), 2);
            mean_grid_statistics = mean(grid_statistics(valid_grids, :), 1);
            ts_rows = IDX_TS_FIRST:n_methods;
            Results_Table.MeanGridX1 = nan(n_methods, 1);
            Results_Table.MeanGridX2 = nan(n_methods, 1);
            Results_Table.MeanGridCells = nan(n_methods, 1);
            Results_Table.MeanEmptyCellsPct = nan(n_methods, 1);
            Results_Table.N_grid = nan(n_methods, 1);
            Results_Table{ts_rows, {'MeanGridX1', 'MeanGridX2', ...
                'MeanGridCells', 'MeanEmptyCellsPct'}} = ...
                repmat(mean_grid_statistics, numel(ts_rows), 1);
            Results_Table.N_grid(ts_rows) = nnz(valid_grids);

            fprintf('\n==== ERRORS: Train Ratio %.2f | Sampling "%s" | Dataset "%s" (N_MC = %d) ====\n', ...
                current_ratio, current_sampling, dataset_tag, n_mc_iterations);
            disp(Results_Table);

            if verify_guarantees
                pass_rate  = 100 * mean(guarantee_pass, 3, 'omitnan');
                quant_mean = mean(guarantee_quant, 3, 'omitnan');

                n_eval = squeeze(sum(~isnan(guarantee_pass(:, 1, :)), 3));

                pass_cols = cell(1, n_guarantees);
                for g = 1:n_guarantees
                    pass_cols{g} = compose('%.1f%%', pass_rate(:, g));
                end
                Guarantee_Table = table(pass_cols{:}, n_eval, ...
                    'RowNames', cellstr(ts_variant_names), ...
                    'VariableNames', [reshape(cellstr(guarantee_names), 1, []), {'N_eval'}]);

                fprintf('\n==== FORMAL GUARANTEES (approval rate over %d replications) ====\n', n_mc_iterations);
                disp(Guarantee_Table);

                Guarantee_Quant_Table = table( ...
                    compose('%.3f', quant_mean(:, 1)), ...
                    compose('%.3f', quant_mean(:, 2)), ...
                    compose('%.4g', quant_mean(:, 3)), ...
                    compose('%.4g', quant_mean(:, 4)), ...
                    compose('%.3e', quant_mean(:, 5)), ...
                    'RowNames', cellstr(ts_variant_names), ...
                    'VariableNames', {'RobustTightness', 'TruncTightness', ...
                                      'Term1_RasterImput', 'Term2_HOSVDTrunc', 'Term3_BasisRecon'});

                fprintf('\n==== FORMAL GUARANTEES (mean quantitative values) ====\n');
                disp(Guarantee_Quant_Table);
            end

            %% Numerical exports
            if export_results
                tag = sprintf('%s_ratio%02d_%s', dataset_tag, round(current_ratio*100), current_sampling);
                writetable(Results_Table, fullfile(results_dir, sprintf('errors_%s.csv', tag)), ...
                    'WriteRowNames', true);
                if verify_guarantees
                    writetable(Guarantee_Table, fullfile(results_dir, sprintf('guarantees_%s.csv', tag)), ...
                        'WriteRowNames', true);
                    writetable(Guarantee_Quant_Table, fullfile(results_dir, sprintf('guarantees_quant_%s.csv', tag)), ...
                        'WriteRowNames', true);
                end
            end

            % Store raw results for reproducibility
            all_results(end+1) = struct( ...
                'dataset', dataset_name, ...
                'month', current_month, ...
                'train_ratio', current_ratio, ...
                'sampling', char(current_sampling), ...
                'convexification', char(convexification_type), ...
                'convexification_diagnostics', {convexification_diagnostics}, ...
                'spatial_options',spatial_options,'kriging_models',{kriging_models}, ...
                'accumulated_errors', accumulated_errors, ...
                'grid_statistics', grid_statistics, ...
                'guarantee_pass', guarantee_pass, ...
                'guarantee_quant', guarantee_quant); %#ok<SAGROW>

        end
    end

    if export_results
        mat_path = fullfile(results_dir, sprintf('mc_raw_%s.mat', dataset_tag));
        save(mat_path, 'all_results', 'model_names_final', 'metric_names', ...
            'guarantee_names', 'ts_variant_names', 'n_mc_iterations', ...
            'train_ratios_list', 'sampling_methods_list', 'dataset_name', ...
            'current_month', 'dataset_tag', 'run_tag', ...
            'spatial_options','kriging_options','idw_params', ...
            'chosen_datasets', 'chosen_months', 'n_samples_per_input', 'grid_stat_names', ...
            'convexification_type', 'convexification_tolerance', 'cno_options');
        fprintf('\nRaw results saved to: %s\n', mat_path);
    end

end

fprintf('\nMonte Carlo simulation successfully finished.\n');

%% Local metric helpers
function metrics_row = compute_metrics(y_true, y_pred, number_rules)
% COMPUTE_METRICS Return error metrics and model complexity in fixed order.
%   y_true and y_pred are equally sized observations and predictions;
%   number_rules is the model's rule count. metrics_row contains RE, MAE,
%   RMSE, error standard deviation, maximum absolute error and rule count.
    [~, MAE, RMSE, RE, ~, MaxAE, StdErr] = calculate_erros(y_true, y_pred);
    metrics_row = [RE, MAE, RMSE, StdErr, MaxAE, number_rules];
end

function col = format_mean_std(mean_vec, std_vec)
% FORMAT_MEAN_STD Format corresponding means and standard deviations.
%   mean_vec and std_vec contain corresponding numeric values. col is a
%   cell column of formatted strings, with N/A for missing means.
    n = numel(mean_vec);
    col = cell(n, 1);
    for i = 1:n
        if isnan(mean_vec(i))
            col{i} = 'N/A';
        else
            col{i} = sprintf('%.3f \x00B1 %.3f', mean_vec(i), std_vec(i));
        end
    end
end
