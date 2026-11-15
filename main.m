%% Single-run interpolation comparison
% Run from the repository root. Configure the dataset, sampling, imputation,
% and model settings below. Results and diagnostics remain in the workspace;
% figures are created when global_disable_plots is false.
addpath('./tp-model-transformation/');
addpath('./auxiliary/');
addpath('./interpolation-algorithm/');
addpath('./datasets/');
addpath('./model-analyzes/');
addpath('./formal-guarantees/');

clearvars; close all; clc;

%% Experiment settings
rng(1)

global_disable_plots = false;

n_samples_per_input = 20;
avaliable_datasets = [
    "1 - linear_sine"
    "2 - exp_2d"
    "3 - franke"
    "4 - peaks"
    "5 - sinc"
    "6 - gauss_mixture"
    "7 - saddle"
    "8 - wind_temperature"
];
chosen_dataset = avaliable_datasets(8);
chosen_dataset = strsplit(chosen_dataset, "- ");
chosen_dataset = chosen_dataset{2};
chosen_mount = 2;   % NOAA month index (1=January, ..., 12=December).

% Dataset-specific variogram and interpolation settings.
spatial_options = spatial_method_options(chosen_dataset);
kriging_options = struct('type',spatial_options.kriging_type, ...
    'fit_variogram',spatial_options.kriging_fit);
grid_options = {'KrigingModel',kriging_options};

% Sampling strategy and TS grid imputation.
sampling_methods = ["random", "grid"];
sampling_method = sampling_methods(2);
fill_methods = [
    "1 - global_mean"
    "2 - interp"
    "3 - kriging"
    "4 - knn"
    "5 - tensor_completion"
    "6 - none"
];
fill_method = fill_methods(2);
fill_method = strsplit(fill_method, "- ");
fill_method = fill_method{2};

% Fraction of observations requested for training.
train_ratio = 0.5;

%% Reference dataset
Dataset = get_dataset(n_samples_per_input, chosen_dataset, chosen_mount);

if (~global_disable_plots)
    figure; %#ok<*UNRCH>
    x1 = unique(Dataset.X(:,1));
    x2 = unique(Dataset.X(:,2));
    [X1, X2] = meshgrid(x1, x2);
    Y_Grid = reshape(Dataset.Y, size(X1));

    contourf(X1, X2, Y_Grid, 20, 'LineColor', 'none');
    xlabel('$x_{1}$', 'Interpreter','latex'); ylabel('$x_{2}$', 'Interpreter','latex'); colorbar;
end

%% Training and test partition
[Train_Dataset, Test_Dataset] = split_dataset(Dataset, train_ratio, chosen_dataset, sampling_method);

if (~global_disable_plots)
    figure;
    contourf(X1, X2, Y_Grid, 20, 'LineColor', 'none');
    hold on;
    scatter(Train_Dataset.X(:,1), Train_Dataset.X(:,2), 15, 'k', 'filled');
    xlabel('$x_{1}$', 'Interpreter','latex');
    ylabel('$x_{2}$', 'Interpreter','latex');
    title('Spatial Visualization of Training Points');
    colorbar; grid on;
    hold off;
end

%% Training grid
if strcmpi(sampling_method, "grid")
    grid_size = [length(unique(Train_Dataset.X(:,1))) length(unique(Train_Dataset.X(:,2)))];
else
    suggested_grid_size = estimate_grid_size(Train_Dataset.X);
    disp("Suggested grid: "); disp(suggested_grid_size);

    grid_size = suggested_grid_size;
end
disp("Used grid: "); disp(grid_size);

% split_dataset can fall back to Random when Grid is not applicable.
preserve_sample_grid = strcmpi(sampling_method,'grid') && ...
    size(unique(Train_Dataset.X,'rows'),1) == prod(grid_size);
Train_Dataset_Grid = create_grid_from_samples(Train_Dataset, grid_size, fill_method, ...
    grid_options{:},'PreserveSampleGrid',preserve_sample_grid);

if (~global_disable_plots)
    plot_grid_2d(Train_Dataset, Train_Dataset_Grid);
end

%% Convex TP model training
convexification_tolerance = 1e-6;
convexification_type = "CNO"; % "SNNN", "CNO", or "RNO-INO"
cno_options = struct(); % Used only for CNO.
singular_value_tolerance = 1e-2; % Keep: singular_value >= singular_value_tolerance * max(singular_values)
[~, ~, S_rno, U_rno, ~, convexification_info] = train_ts_model(Train_Dataset_Grid, ...
    singular_value_tolerance, convexification_tolerance, ...
    'Convexification', convexification_type, 'CNOOptions', cno_options);
fprintf('Convexification: %s | relative preservation error: %.3e\n', ...
    convexification_info.type, convexification_info.reconstruction_error);

%% TP predictions and membership diagnostics
Estimated_Dataset = estimate_data(Test_Dataset, Train_Dataset_Grid, U_rno, S_rno);

grid_x1 = unique(Train_Dataset_Grid.X(:,1));
grid_x2 = unique(Train_Dataset_Grid.X(:,2));

if (~global_disable_plots)
    mf_report = analyze_membership_functions(grid_x1, grid_x2, U_rno, ...
        'Title', sprintf('%s, fill=%s, rho=%.2f', chosen_dataset, fill_method, train_ratio)); %#ok<NASGU>
end

I = size(S_rno);
output_order = size(Estimated_Dataset.Y(1, 1), 2) - 1;
sizes = I(1:end-output_order);
number_rules = prod(sizes);
fprintf('Number of rules: %d\n', number_rules);

%% Structural checks and numerical bounds
formal_guarantee_report = run_formal_guarantee_checks( ...
    Dataset, Train_Dataset_Grid, Test_Dataset, S_rno, U_rno, singular_value_tolerance);

%% Full-reference reconstruction
disp("================ FINAL PLOT: REAL vs TS ================")

% Full-reference errors include training points and are not test-set metrics.
Estimated_Full_Dataset = estimate_data(Dataset, Train_Dataset_Grid, U_rno, S_rno);

x1_unique = unique(Dataset.X(:,1));
x2_unique = unique(Dataset.X(:,2));

if length(x1_unique) * length(x2_unique) == length(Dataset.Y)
    [XX1, XX2] = meshgrid(x1_unique, x2_unique);
    Z_Real = reshape(Dataset.Y, length(x2_unique), length(x1_unique));
    Z_Est  = reshape(Estimated_Full_Dataset.Y, length(x2_unique), length(x1_unique));
else
    % Rasterize irregular coordinates for visualization only.
    res_plot = 100;
    x1_lin = linspace(min(Dataset.X(:,1)), max(Dataset.X(:,1)), res_plot);
    x2_lin = linspace(min(Dataset.X(:,2)), max(Dataset.X(:,2)), res_plot);
    [XX1, XX2] = meshgrid(x1_lin, x2_lin);

    Z_Real = griddata(Dataset.X(:,1), Dataset.X(:,2), Dataset.Y, XX1, XX2, 'natural');
    Z_Est  = griddata(Dataset.X(:,1), Dataset.X(:,2), Estimated_Full_Dataset.Y, XX1, XX2, 'natural');
end

if (~global_disable_plots)
    figure;

    % Use a common color scale for reference and reconstruction.
    z_min = min([Z_Real(:); Z_Est(:)]);
    z_max = max([Z_Real(:); Z_Est(:)]);

    subplot(1, 2, 1);
    contourf(XX1, XX2, Z_Real, 30, 'LineColor', 'none');
    title('Original Dataset (Real)', 'FontSize', 12);
    xlabel('$x_{1}$', 'Interpreter','latex');
    ylabel('$x_{2}$', 'Interpreter','latex');
    colorbar;
    clim([z_min z_max]);
    axis tight; axis square;

    subplot(1, 2, 2);
    contourf(XX1, XX2, Z_Est, 30, 'LineColor', 'none');
    title('TS Model Reconstruction', 'FontSize', 8);
    xlabel('$x_{1}$', 'Interpreter','latex');
    ylabel('$x_{2}$', 'Interpreter','latex');
    colorbar;
    clim([z_min z_max]);
    axis tight; axis square;
end

rmse_global = sqrt(mean((Dataset.Y - Estimated_Full_Dataset.Y).^2));
re_global = norm(Dataset.Y - Estimated_Full_Dataset.Y) / norm(Dataset.Y);

fprintf('\nGlobal Reconstruction Error (Full Dataset):\n');
fprintf('RMSE: %.4f\n', rmse_global);
fprintf('RE:   %.4f (%.2f%%)\n', re_global, re_global*100);


%% Kriging predictions
[y_pred_kriging, ~, kriging_var_model] = kriging_predict(Train_Dataset, Test_Dataset,kriging_options);
fprintf('%s variogram: nugget=%.4g, sill=%.4g, range=%.4g\n', ...
    kriging_var_model.type, ...
    kriging_var_model.nugget, kriging_var_model.sill, kriging_var_model.range);

%% Global inverse-distance weighting
idw_params.power = 2;
idw_params.k = []; % Global IDW: use all distinct training locations.
y_pred_idw = idw_predict(Train_Dataset, Test_Dataset, idw_params);

%% Scattered interpolation
y_pred_interp = interpolation_predict(Train_Dataset, Test_Dataset, ...
    'Method',spatial_options.scattered_method,'Extrapolation','nearest');

%% Neural-network predictions
y_pred_nn = neural_network_predict(...
    Train_Dataset, ...
    Test_Dataset, ...
    'HiddenNeurons', 20, ...
    'TrainFcn', 'trainbr', ... % Bayesian regularization.
    'ShowWindow', false ...
);

%% Test-set error comparison
[~, MAE_ts, RMSE_ts, RE_ts, ~, MaxAE_ts, StdErr_ts] = calculate_erros(Test_Dataset.Y, Estimated_Dataset.Y);

[~, MAE_kriging, RMSE_kriging, RE_kriging, ~, MaxAE_kriging, StdErr_kriging] = calculate_erros(Test_Dataset.Y, y_pred_kriging);

[~, MAE_idw, RMSE_idw, RE_idw, ~, MaxAE_idw, StdErr_idw] = calculate_erros(Test_Dataset.Y, y_pred_idw);

[~, MAE_interp, RMSE_interp, RE_interp, ~, MaxAE_interp, StdErr_interp] = calculate_erros(Test_Dataset.Y, y_pred_interp);

[~, MAE_nn, RMSE_nn, RE_nn, ~, MaxAE_nn, StdErr_nn] = calculate_erros(Test_Dataset.Y, y_pred_nn);

Model_Names = {'TP Model Transformation'; 'Kriging'; 'IDW'; 'Scattered Interp'; 'Neural Net'};
RE_vals     = [RE_ts; RE_kriging; RE_idw; RE_interp; RE_nn];
MAE_vals    = [MAE_ts; MAE_kriging; MAE_idw; MAE_interp; MAE_nn];
RMSE_vals   = [RMSE_ts; RMSE_kriging; RMSE_idw; RMSE_interp; RMSE_nn];
MaxAE_vals  = [MaxAE_ts; MaxAE_kriging; MaxAE_idw; MaxAE_interp; MaxAE_nn];
StdErr_vals = [StdErr_ts; StdErr_kriging; StdErr_idw; StdErr_interp; StdErr_nn];
Rules_vals  = [number_rules; NaN; NaN; NaN; NaN]; % model complexity

Results_Table = table(...
    compose('%.4f', RE_vals), ...
    compose('%.4f', MAE_vals), ...
    compose('%.4f', RMSE_vals), ...
    compose('%.4f', StdErr_vals), ...
    compose('%.4f', MaxAE_vals), ...
    Rules_vals, ...
    'RowNames', Model_Names, ...
    'VariableNames', {'RE', 'MAE', 'RMSE', 'StdErr', 'MaxAE', 'Rules'});

disp("================ ERROR COMPARISON ================")
disp(Results_Table);

if (~global_disable_plots)
    figure;

    bar_data = [RE_vals, MAE_vals, RMSE_vals];
    b = bar(bar_data);

    set(gca, 'XTickLabel', Model_Names);
    legend({'RE', 'MAE', 'RMSE'}, 'Location', 'NorthWest');
    ylabel('Error Magnitude');
    title('Model Performance Comparison');
    grid on;
    ylim([0 max(bar_data(:)) * 1.15]);

    for i = 1:length(b)
        xtips = b(i).XEndPoints;
        ytips = b(i).YEndPoints;

        labels = string(round(ytips, 4));

        text(xtips, ytips, labels, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontSize', 8, ...
            'FontWeight', 'bold');
    end
end
