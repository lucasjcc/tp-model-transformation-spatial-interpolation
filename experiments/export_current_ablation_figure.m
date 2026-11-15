function export_current_ablation_figure
% EXPORT_CURRENT_ABLATION_FIGURE Export the B/TP paired-comparison figure.
%   Takes no arguments and returns no output. Requires the full current
%   analyze_current_ablation results. Checks summaries against paired records,
%   writes r2_ablation.pdf to article/manuscrito/figures, and saves a PNG and
%   audit CSVs under results/reviewer1/cno. Existing exports are overwritten.

    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root,'auxiliary'));
    source_dir = project_results_path('reviewer1','cno','',true);
    output_dir = project_results_path('reviewer1','cno');
    if ~isfolder(output_dir), mkdir(output_dir); end
    data = readtable(fullfile(source_dir,'ablation_summary.csv'),'TextType','string');
    assert(height(data) == 30 && all(data.n == 50) && all(data.csv_summary_matches));
    tags = ["franke","peaks","sinc","gauss_mixture", ...
        "wind_temperature_month01","wind_temperature_month02","wind_temperature_month03", ...
        "wind_temperature_month04","wind_temperature_month05","wind_temperature_month06"];
    labels = ["Franke","Peaks","Sinc","Gaussian", ...
        "NOAA Jan","NOAA Feb","NOAA Mar","NOAA Apr","NOAA May","NOAA Jun"];
    ratios = [.1,.3,.5];
    fig = figure('Visible','off','Color','white','Units','centimeters','Position',[1,1,18.5,11.5]);
    cleanup = onCleanup(@() close(fig));
    layout = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
    plotted_rows = cell(30,7);
    raw_rows = cell(1500,8);
    for j = 1:3
        ax = nexttile(layout);
        hold(ax,'on');
        interval_limits = zeros(10,2);
        for d = 1:10
            entry = data(data.dataset == tags(d) & abs(data.ratio-ratios(j)) < 1e-8,:);
            assert(height(entry) == 1 && entry.mean_before > 0);
            raw = load(fullfile(source_dir,sprintf('ablation_%s_ratio%d.mat',tags(d),round(100*ratios(j)))));
            delta = raw.rmse(:,2)-raw.rmse(:,1);
            assert(isequal(size(raw.rmse),[50,3]));
            assert(isequal(raw.seeds,(1:50)'+50*(j-1)));
            assert(abs(mean(delta)-entry.mean_delta) < 1e-13);
            assert(abs(mean(raw.rmse(:,1))-entry.mean_before) < 1e-13);
            se = std(delta)*sqrt(1/50+mean(raw.counts(:,2)./raw.counts(:,1)));
            ci = mean(delta)+[-1,1]*tinv(.975,49)*se;
            assert(max(abs(ci-[entry.ci_low,entry.ci_high])) < 1e-13);
            factor = 100/entry.mean_before;
            interval_limits(d,:) = factor*[entry.ci_low,entry.ci_high];
            color = [0.0,0.45,0.70];
            marker = 'o';
            if entry.reject_005
                color = [0.80,0.20,0.35];
                marker = 's';
            end
            plot(ax,interval_limits(d,:),[d,d],'Color',color,'LineWidth',1.3);
            plot(ax,factor*entry.mean_delta,d,marker, ...
                'Color',color,'MarkerFaceColor',color,'MarkerSize',4.5);
            plotted_rows((j-1)*10+d,:) = {tags(d),ratios(j),entry.mean_before, ...
                factor*entry.mean_delta,interval_limits(d,1),interval_limits(d,2),entry.reject_005};
            for k = 1:50
                raw_rows(((j-1)*10+d-1)*50+k,:) = {tags(d),ratios(j),raw.seeds(k), ...
                    raw.counts(k,1),raw.counts(k,2),raw.rmse(k,1),raw.rmse(k,2),raw.rmse(k,3)};
            end
        end
        xline(ax,0,'--','Color',[0.4,0.4,0.4],'LineWidth',0.8);
        limits = [min([interval_limits(:);0]),max([interval_limits(:);0])];
        padding = 0.08*diff(limits);
        xlim(ax,limits+[-padding,padding]);
        ax.YDir = 'reverse';
        ax.YTick = 1:10;
        ylim(ax,[0.4,10.6]);
        if j == 1
            ax.YTickLabel = labels;
        else
            ax.YTickLabel = [];
        end
        title(ax,sprintf('(%c) Training %d%%','a'+j-1,round(100*ratios(j))),'FontWeight','normal');
        grid(ax,'on');
        set(ax,'FontName','Arial','FontSize',10,'LineWidth',0.6, ...
            'Box','on','TickDir','out','GridAlpha',0.12,'GridColor','black', ...
            'Color','white','XColor','black','YColor','black','XMinorGrid','off','YMinorGrid','off');
        set([ax.Title,ax.XLabel,ax.YLabel],'Color','black');
    end
    xlabel(layout,'RMSE difference / reference mean (%)','FontSize',10,'Interpreter','none','Color','black');
    plotted = cell2table(plotted_rows,'VariableNames', ...
        {'dataset','ratio','mean_before','delta_pct','ci_low_pct','ci_high_pct','reject_005'});
    writetable(plotted,fullfile(output_dir,'ablation_plot_data.csv'));
    raw_table = cell2table(raw_rows,'VariableNames', ...
        {'dataset','ratio','seed','n_train','n_test','rmse_before','rmse_tp','rmse_scattered'});
    writetable(raw_table,fullfile(output_dir,'ablation_pairs.csv'));
    exportgraphics(fig,fullfile(root,'article','manuscrito','figures','r2_ablation.pdf'),'ContentType','vector');
    exportgraphics(fig,fullfile(output_dir,'r2_ablation.png'),'Resolution',180);
    fprintf('Exported only r2_ablation: 30 contrasts and marginal corrected intervals checked against raw pairs.\n');
end
