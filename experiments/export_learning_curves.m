function export_learning_curves
% EXPORT_LEARNING_CURVES Export Random-sampling RMSE curves from CNO CSVs.
%   Takes no arguments and returns no output. Run from the repository root.
%   Reads published three-decimal means without refitting. Writes a PDF to
%   article/manuscrito/figures and PNG/audit CSV files under results/reviewer2.
%   Requires 50 valid repetitions per source row; existing exports are overwritten.

    root_dir = pwd;
    addpath(fullfile(root_dir,'auxiliary'));
    source_dir = project_results_path('monte_carlo','cno','random_n50',true);
    figure_dir = fullfile(root_dir, 'article', 'manuscrito', 'figures');
    output_dir = project_results_path('reviewer2','cno');
    tags = ["franke", "peaks", "sinc", "gauss_mixture", ...
        "wind_temperature_month01", "wind_temperature_month02", ...
        "wind_temperature_month03", "wind_temperature_month04", ...
        "wind_temperature_month05", "wind_temperature_month06"];
    labels = ["Franke", "Peaks", "Sinc", "Gaussian", ...
        "NOAA Jan", "NOAA Feb", "NOAA Mar", "NOAA Apr", "NOAA May", "NOAA Jun"];
    ratios = [0.1, 0.3, 0.5];
    methods = ["Kriging", "IDW", "Scattered", "Neural Net", "TS (Interp)"];
    display_methods = ["Kriging", "IDW", "Scattered", "NN", "TS-I"];
    colors = [0.0, 0.45, 0.70; 0.0, 0.60, 0.50; 0.25, 0.25, 0.25; ...
        0.80, 0.45, 0.0; 0.80, 0.20, 0.35];
    markers = {'o', '^', 's', 'd', 'v'};
    means = zeros(10, 3, 5);
    curve_rows = cell(150, 5);
    row_index = 0;
    for d = 1:numel(tags)
        for j = 1:numel(ratios)
            path = fullfile(source_dir, sprintf('errors_%s_ratio%02d_random.csv', ...
                tags(d), round(100*ratios(j))));
            summary = readtable(path, 'TextType', 'string');
            for m = 1:numel(methods)
                method_row = summary(summary.Row == methods(m), :);
                assert(height(method_row) == 1 && method_row.N_ok == 50, ...
                    'Expected one method row with 50 valid runs: %s, %s', path, methods(m));
                tokens = regexp(char(method_row.RMSE), ...
                    '^(\d+\.\d{3})\s*±\s*(\d+\.\d{3})$', 'tokens', 'once');
                assert(~isempty(tokens), 'Invalid published RMSE: %s, %s', path, methods(m));
                means(d, j, m) = str2double(tokens{1});
                assert(isfinite(means(d, j, m)) && means(d, j, m) > 0, ...
                    'Logarithmic axes require positive finite RMSEs.');
                row_index = row_index+1;
                curve_rows(row_index, :) = {tags(d), ratios(j), display_methods(m), ...
                    means(d, j, m), 50};
            end
        end
    end

    fig = figure('Visible', 'off', 'Color', 'white', ...
        'Units', 'centimeters', 'Position', [1, 1, 18.5, 22]);
    cleanup = onCleanup(@() close(fig));
    layout = tiledlayout(fig, 5, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    handles = gobjects(5, 1);
    for d = 1:numel(tags)
        ax = nexttile(layout);
        hold(ax, 'on');
        for m = 1:numel(methods)
            handles(m) = plot(ax, ratios*100, means(d, :, m), ...
                'Color', colors(m, :), 'Marker', markers{m}, ...
                'LineWidth', 1.1, 'MarkerSize', 4);
            assert(isequal(handles(m).YData, means(d, :, m)), ...
                'Plotted points differ from the CSV means.');
        end
        ax.YScale = 'log';
        ax.XTick = ratios*100;
        xlim(ax, [7, 53]);
        panel_values = reshape(means(d, :, :), [], 1);
        ylim(ax, [0.9*min(panel_values), 1.1*max(panel_values)]);
        if max(panel_values)/min(panel_values) >= 10
            ax.YTick = 10.^(ceil(log10(0.9*min(panel_values))): ...
                floor(log10(1.1*max(panel_values))));
        end
        title(ax, sprintf('(%c) %s', 'a'+d-1, labels(d)), 'FontWeight', 'normal');
        ylabel(ax, 'Mean RMSE');
        if d >= 9
            xlabel(ax, 'Training (%)');
        end
        grid(ax, 'on');
        set(ax, 'FontName', 'Arial', 'FontSize', 10, 'LineWidth', 0.6, ...
            'Box', 'on', 'TickDir', 'out', 'GridAlpha', 0.12, 'GridColor', 'black', ...
            'Color', 'white', 'XColor', 'black', 'YColor', 'black', 'ZColor', 'black', ...
            'XMinorGrid', 'off', 'YMinorGrid', 'off', 'ZMinorGrid', 'off');
        set([ax.Title, ax.XLabel, ax.YLabel, ax.ZLabel], 'Color', 'black');
    end
    leg = legend(handles, display_methods, 'NumColumns', 5, 'FontSize', 10, ...
        'Color', 'white', 'TextColor', 'black', 'EdgeColor', [0.5, 0.5, 0.5]);
    leg.Layout.Tile = 'south';

    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    curve_table = cell2table(curve_rows, 'VariableNames', ...
        {'dataset', 'ratio', 'method', 'mean_rmse_csv', 'n'});
    writetable(curve_table, fullfile(output_dir, 'learning_curve_data.csv'));
    exportgraphics(fig, fullfile(figure_dir, 'r2_learning_curves.pdf'), 'ContentType', 'vector');
    exportgraphics(fig, fullfile(output_dir, 'r2_learning_curves.png'), 'Resolution', 180);
    fprintf('Exported only r2_learning_curves: 150 CSV means, 30 configurations, N_ok=50.\n');
end
