function report = analyze_membership_functions(grid_x1, grid_x2, U_rno, varargin)
% ANALYZE_MEMBERSHIP_FUNCTIONS Plot and summarize convex TP membership functions.
%   grid_x1 and grid_x2 are node vectors; U_rno contains their weight matrices.
%   Optional Title labels the figure; SavePath requests figure export.
%   report contains per-mode rule counts, linear-fit R^2, normalized roughness,
%   their means, and a heuristic sampling classification. These shape-based
%   labels are not a certificate of sampling adequacy or predictive accuracy.

    p = inputParser;
    addParameter(p, 'Title', '', @(s) ischar(s) || isstring(s));
    addParameter(p, 'SavePath', '', @(s) ischar(s) || isstring(s));
    parse(p, varargin{:});

    dims = {grid_x1(:), grid_x2(:)};
    dim_labels = {'x_1', 'x_2'};
    U = {U_rno{1}, U_rno{2}};

    figure;

    report = struct('label', {}, 'n_rules', {}, 'linearity_r2', {}, ...
        'roughness', {}, 'mean_linearity_r2', {}, 'mean_roughness', {}, 'verdict', {});

    for k = 1:2
        subplot(2, 1, k);
        x = dims{k};
        Uk = U{k};
        n_rules = size(Uk, 2);

        cmap = lines(max(n_rules, 1));
        hold on;
        legend_entries = cell(1, n_rules);
        lin_r2 = zeros(1, n_rules);
        roughness = zeros(1, n_rules);

        for i = 1:n_rules
            u = Uk(:, i);
            plot(x, u, 'LineWidth', 1.6, 'Color', cmap(i, :));
            legend_entries{i} = sprintf('Rule %d', i);

            % Linearity: R^2 of the best-fit line through (x, u)
            coeffs = polyfit(x, u, 1);
            u_fit = polyval(coeffs, x);
            ss_res = sum((u - u_fit).^2);
            ss_tot = sum((u - mean(u)).^2);
            if ss_tot < eps
                lin_r2(i) = 1; % (near-)constant function: trivially flat/linear
            else
                lin_r2(i) = max(0, 1 - ss_res / ss_tot);
            end

            % Roughness: RMS of the normalized second finite difference
            if numel(u) >= 3
                d2 = diff(u, 2);
                span = max(u) - min(u);
                roughness(i) = sqrt(mean(d2.^2)) / (span + eps);
            else
                roughness(i) = 0;
            end
        end
        hold off;

        grid on; box on;
        xlim([min(x), max(x)]);
        ylim([-0.05, 1.05]);
        xlabel(sprintf('$%s$', dim_labels{k}), 'Interpreter', 'latex', 'FontSize', 12);
        ylabel('Membership degree $w(x)$', 'Interpreter', 'latex', 'FontSize', 12);

        mean_r2 = mean(lin_r2);
        mean_rough = mean(roughness);
        verdict = classify_sampling(mean_r2, mean_rough);

        title_str = sprintf('Mode %d (%s): %d rules -- mean R^2=%.3f, mean roughness=%.3f -> %s', ...
            k, dim_labels{k}, n_rules, mean_r2, mean_rough, verdict);
        title(title_str, 'FontSize', 11, 'Interpreter', 'none');

        if n_rules >= 1 && n_rules <= 15
            legend(legend_entries, 'Location', 'eastoutside', 'FontSize', 8);
        end

        report(k).label = dim_labels{k};
        report(k).n_rules = n_rules;
        report(k).linearity_r2 = lin_r2;
        report(k).roughness = roughness;
        report(k).mean_linearity_r2 = mean_r2;
        report(k).mean_roughness = mean_rough;
        report(k).verdict = verdict;

        fprintf('[analyze_membership_functions] Mode %d (%s): mean R^2=%.3f, mean roughness=%.3f -> %s\n', ...
            k, dim_labels{k}, mean_r2, mean_rough, verdict);
    end

    fig_title = 'Membership functions -- sampling-density diagnostic';
    if ~isempty(p.Results.Title)
        fig_title = sprintf('%s (%s)', fig_title, p.Results.Title);
    end
    sgtitle(fig_title, 'FontWeight', 'bold', 'Interpreter', 'none');

    if ~isempty(p.Results.SavePath)
        exportgraphics(fig, p.Results.SavePath, 'Resolution', 300);
    end
end

function verdict = classify_sampling(r2, rough)
% CLASSIFY_SAMPLING Assign a heuristic label from mean R^2 and roughness.
%   r2 and rough are scalar diagnostics; verdict is a descriptive character vector.
    if r2 > 0.9 && rough < 0.15
        verdict = 'likely OVERSAMPLED (near-linear -> coarser future sampling likely sufficient)';
    elseif rough > 0.35 || r2 < 0.5
        verdict = 'likely UNDERSAMPLED (irregular/non-smooth -> prioritize more samples here)';
    else
        verdict = 'adequately sampled';
    end
end
