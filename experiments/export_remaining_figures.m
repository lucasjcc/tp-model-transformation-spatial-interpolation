function export_remaining_figures
% EXPORT_REMAINING_FIGURES Export paired-comparison and synthetic-surface figures.
%   Takes no arguments and returns no output. Requires analyze_remaining_materials
%   results. Uses validated paired summaries and 20x20 analytic reference grids.
%   Writes PDFs to article/manuscrito/figures and PNG/CSV/MAT audit files under
%   results/reviewer4/cno. Existing exports are overwritten.

    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root,'auxiliary'));
    source = project_results_path('reviewer4','cno','',true);
    output = project_results_path('reviewer4','cno');
    if ~isfolder(output), mkdir(output); end
    figure_dir = fullfile(root,'article','manuscrito','figures');
    data = readtable(fullfile(source,'paired_summary.csv'),'TextType','string');
    tags = ["franke","peaks","sinc","gauss_mixture", ...
        "wind_temperature_month01","wind_temperature_month02","wind_temperature_month03", ...
        "wind_temperature_month04","wind_temperature_month05","wind_temperature_month06"];
    labels = ["Franke","Peaks","Sinc","Gaussian", ...
        "NOAA Jan","NOAA Feb","NOAA Mar","NOAA Apr","NOAA May","NOAA Jun"];
    assert(height(data)==30 && all(data.n==50) && all(data.csv_summary_matches));
    ratios = [.1,.3,.5];
    fig = new_figure([18.5,11.5]);
    layout = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
    plotted = cell(30,6);
    for j = 1:3
        ax = nexttile(layout); hold(ax,'on');
        limits = zeros(10,2);
        for d = 1:10
            r = data(data.dataset==tags(d) & abs(data.ratio-ratios(j))<1e-12,:);
            assert(height(r)==1 && r.mean_scattered>0 && r.ci_low<=r.mean_delta && r.mean_delta<=r.ci_high);
            scale = 100/r.mean_scattered;
            limits(d,:) = scale*[r.ci_low,r.ci_high];
            color = [0,.45,.70]; marker = 'o';
            if r.reject_005, color = [.8,.2,.35]; marker = 's'; end
            plot(ax,limits(d,:),[d,d],'Color',color,'LineWidth',1.3);
            plot(ax,scale*r.mean_delta,d,marker,'Color',color,'MarkerFaceColor',color,'MarkerSize',4.5);
            plotted((j-1)*10+d,:) = {tags(d),ratios(j),scale*r.mean_delta,limits(d,1),limits(d,2),r.reject_005};
        end
        xline(ax,0,'--','Color',[.4,.4,.4],'LineWidth',.8);
        span = [min([limits(:);0]),max([limits(:);0])];
        padding = .08*diff(span); xlim(ax,span+[-padding,padding]);
        ax.YDir = 'reverse'; ax.YTick = 1:10; ylim(ax,[.4,10.6]);
        if j==1, ax.YTickLabel=labels; else, ax.YTickLabel=[]; end
        title(ax,sprintf('(%c) Training %d%%','a'+j-1,round(100*ratios(j))),'FontWeight','normal');
        grid(ax,'on'); style_axes(ax);
    end
    xlabel(layout,'RMSE difference / reference mean (%)','FontSize',10,'Interpreter','none','Color','black');
    writetable(cell2table(plotted,'VariableNames', ...
        {'dataset','ratio','delta_pct','ci_low_pct','ci_high_pct','reject_005'}),fullfile(output,'paired_plot_data.csv'));
    save_figure(fig,'r2_paired_comparison',figure_dir,output,false);

    fig = new_figure([18.5,13.5]);
    layout = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
    references = struct([]);
    for d = 1:4
        reference = get_dataset(20,tags(d),1);
        assert(size(reference.X,1)==400 && all(isfinite(reference.Y)));
        if d==2
            expected = peaks(reshape(reference.X(:,1),20,20),reshape(reference.X(:,2),20,20));
            assert(max(abs(expected(:)-reference.Y))<1e-12);
        end
        references(d).tag = tags(d);
        references(d).data = reference;
        ax = nexttile(layout);
        surf(ax,reshape(reference.X(:,1),20,20),reshape(reference.X(:,2),20,20), ...
            reshape(reference.Y,20,20),'FaceColor','interp','EdgeColor',[.35,.35,.35],'LineWidth',.15);
        view(ax,42,32); xlabel(ax,'x_1'); ylabel(ax,'x_2'); zlabel(ax,'f(x_1,x_2)');
        title(ax,sprintf('(%c) %s','a'+d-1,labels(d)),'FontWeight','normal');
        colormap(ax,parula(128)); axis(ax,'tight'); grid(ax,'on'); style_axes(ax);
    end
    save(fullfile(output,'synthetic_references.mat'),'references');
    save_figure(fig,'r2_synthetic_surfaces',figure_dir,output,true);
end

function fig = new_figure(size_cm)
% NEW_FIGURE Create a hidden white figure with [width,height] size_cm.
%   fig is the figure handle; dimensions are in centimeters.
    fig = figure('Visible','off','Color','white','Units','centimeters','Position',[1,1,size_cm]);
end

function style_axes(ax)
% STYLE_AXES Apply the shared manuscript appearance to axes handle ax.
%   Updates graphics properties and returns no output.
    set(ax,'FontName','Arial','FontSize',10,'LineWidth',.6,'Box','on','TickDir','out', ...
        'GridAlpha',.12,'GridColor','black','Color','white','XColor','black','YColor','black','ZColor','black', ...
        'XMinorGrid','off','YMinorGrid','off','ZMinorGrid','off');
    set([ax.Title,ax.XLabel,ax.YLabel,ax.ZLabel],'Color','black');
end

function save_figure(fig,name,figure_dir,source,raster)
% SAVE_FIGURE Export figure handle fig and close it.
%   name is the file stem; figure_dir receives the PDF and source the PNG.
%   raster selects image versus vector PDF content. Returns no output.
    if raster
        exportgraphics(fig,fullfile(figure_dir,[name,'.pdf']),'ContentType','image','Resolution',600);
    else
        exportgraphics(fig,fullfile(figure_dir,[name,'.pdf']),'ContentType','vector');
    end
    exportgraphics(fig,fullfile(source,[name,'.png']),'Resolution',180);
    close(fig);
end
