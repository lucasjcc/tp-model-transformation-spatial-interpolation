function export_reviewer3_materials(source_dir, table_dir)
% EXPORT_REVIEWER3_MATERIALS Export prediction maps and noise-study summaries.
%   source_dir selects a completed 50-repetition noise study (default
%   results/reviewer3/cno, with archived references as a fallback);
%   table_dir receives LaTeX (default manuscript tables).
%   Requires analyze_reviewer3_noise results. Returns no output; writes figures
%   to article/manuscrito/figures and audit CSV/JSON files under results/reviewer3/cno.
%   Existing exports are overwritten. Fixed visual examples use seed 51,
%   no added noise, and Franke/NOAA January, with shared scales within each case.

    root_dir = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root_dir,'auxiliary'));
    if nargin < 1
        source_dir = project_results_path('reviewer3','cno','',true);
    end
    output_dir = project_results_path('reviewer3','cno');
    if ~isfolder(output_dir), mkdir(output_dir); end
    figure_dir = fullfile(root_dir, 'article', 'manuscrito', 'figures');
    if nargin < 2
        table_dir = fullfile(root_dir, 'article', 'manuscrito', 'tables');
    end
    summary = readtable(fullfile(source_dir,'noise_summary.csv'), 'TextType','string');
    tags = ["franke", "peaks", "sinc", "gauss_mixture", ...
        "wind_temperature_month01", "wind_temperature_month02", ...
        "wind_temperature_month03", "wind_temperature_month04", ...
        "wind_temperature_month05", "wind_temperature_month06"];
    labels = ["Franke", "Peaks", "Sinc", "Gaussian", ...
        "NOAA Jan", "NOAA Feb", "NOAA Mar", "NOAA Apr", "NOAA May", "NOAA Jun"];
    methods = ["TS-G", "TS-I", "TS-T", "Scattered", "B"];
    levels = [0,.01,.05,.1,.2];
    assert(height(summary)==250 && all(summary.n==50));
    assert(~any(summary.fallbacks));
    max_cno_residual = 0;
    max_baseline_error = 0;
    warning_count = 0;
    pair_rows = cell(12500,7);
    pair_index = 0;
    rank_changes = zeros(10,3);
    for d = 1:10
        raw = load(fullfile(source_dir,sprintf('noise_%s.mat',tags(d))));
        assert(isequal(size(raw.rmse),[50,5,5]) && isequal(raw.seeds,(51:100)'));
        assert(isequal(raw.noise_levels,levels));
        assert(all(isfinite(raw.rmse),'all'));
        assert(all(raw.prediction_change(:,1,:)==0,'all'));
        max_cno_residual = max(max_cno_residual,max(raw.cno_difference));
        max_baseline_error = max(max_baseline_error,raw.metadata.zero_noise_max_reproduction_error);
        warning_count = warning_count+nnz(strlength(raw.warning_messages)>0);
        for m = 1:3
            rank_changes(d,m) = sum(any(raw.ranks(:,5,m,:)~=raw.ranks(:,1,m,:),4));
        end
        for j = 1:5
            for m = 1:5
                row = summary(summary.dataset==tags(d) & summary.method==methods(m) & ...
                    abs(summary.noise_fraction-levels(j))<1e-12,:);
                assert(height(row)==1);
                assert(abs(row.mean_rmse-mean(raw.rmse(:,j,m)))<1e-12);
                assert(abs(row.sd_rmse-std(raw.rmse(:,j,m)))<1e-12);
                assert(abs(row.mean_delta-mean(raw.rmse(:,j,m)-raw.rmse(:,1,m)))<1e-12);
                for k = 1:50
                    pair_index = pair_index+1;
                    pair_rows(pair_index,:) = {tags(d),raw.seeds(k),levels(j),methods(m), ...
                        raw.rmse(k,j,m),raw.prediction_change(k,j,m),raw.noise_sd(k,j)};
                end
            end
        end
    end
    pairs = cell2table(pair_rows,'VariableNames', ...
        {'dataset','seed','noise_fraction','method','rmse','prediction_shift','noise_sd'});
    writetable(pairs,fullfile(output_dir,'noise_observations.csv'));
    rank_table = array2table(rank_changes,'VariableNames',{'TS_G','TS_I','TS_T'});
    rank_table = addvars(rank_table,tags','Before',1,'NewVariableNames','dataset');
    writetable(rank_table,fullfile(output_dir,'rank_changes_at20pct.csv'));
    franke = load(fullfile(source_dir,'snapshot_franke.mat'));
    noaa = load(fullfile(source_dir,'snapshot_wind_temperature_month01.mat'));
    assert(franke.seed==51 && noaa.seed==51);
    intermediate_figure(franke,figure_dir,output_dir);
    comparison_figure(franke,'Franke','r3_comparison_franke',figure_dir,output_dir);
    comparison_figure(noaa,'NOAA January','r3_comparison_noaa',figure_dir,output_dir);

    fig = new_figure([18.5,22]);
    layout = tiledlayout(fig,5,2,'TileSpacing','compact','Padding','compact');
    colors = [.85,.55,0; .80,.20,.35; .50,.30,.75; .25,.25,.25; 0,.55,.80];
    markers = {'o','v','^','s','d'};
    handles = gobjects(5,1);
    for d = 1:10
        ax = nexttile(layout); hold(ax,'on');
        all_means = zeros(5,5);
        for m = 1:5
            for j = 1:5
                row = summary(summary.dataset==tags(d) & summary.method==methods(m) & ...
                    abs(summary.noise_fraction-levels(j))<1e-12,:);
                all_means(j,m) = row.mean_rmse;
            end
            line_style = '-';
            if m==5, line_style='--'; end
            handles(m) = plot(ax,100*levels,all_means(:,m),'Color',colors(m,:), ...
                'Marker',markers{m},'LineStyle',line_style,'LineWidth',1.1,'MarkerSize',4);
        end
        xlim(ax,[-1,21]); ax.XTick=[0,5,10,20];
        lo = min(all_means,[],'all'); hi = max(all_means,[],'all');
        padding = max(.08*(hi-lo),.01*hi);
        ylim(ax,[max(0,lo-padding),hi+padding]);
        title(ax,sprintf('(%c) %s','a'+d-1,labels(d)),'FontWeight','normal');
        ylabel(ax,'Mean RMSE');
        if d>=9, xlabel(ax,'Added noise SD (%)'); end
        style_axes(ax,10); grid(ax,'on');
    end
    leg = legend(handles,methods,'NumColumns',5,'FontSize',10, ...
        'Color','white','TextColor','black','EdgeColor',[.5,.5,.5]);
    leg.Layout.Tile='south';
    save_figure(fig,'r3_noise_sensitivity',figure_dir,output_dir);
    write_noise_table(summary,tags,labels,fullfile(table_dir,'reviewer3_noise.tex'));
    checks = struct('matlab_version',version,'n_cases',10,'n_partitions_per_case',50, ...
        'n_noise_levels',5,'n_methods',5,'n_summary_rows',height(summary), ...
        'n_cno_checks',30,'max_cno_prediction_residual',max_cno_residual, ...
        'max_zero_noise_reproduction_error',max_baseline_error, ...
        'fallback_count',sum(summary.fallbacks), 'warning_count',warning_count, ...
        'checks_passed',true);
    visual_data={franke,noaa};
    examples=struct([]);
    map_rows = cell(800,11);
    map_index = 0;
    for d=1:2
        data=visual_data{d};
        is_test = false(numel(data.dataset.Y),1);
        is_test(data.test_indices)=true;
        for k=1:numel(data.dataset.Y)
            map_index=map_index+1;
            map_rows(map_index,:)={tags(1+4*(d-1)),k,data.dataset.X(k,1),data.dataset.X(k,2), ...
                data.dataset.Y(k),is_test(k),data.predictions(k,1),data.predictions(k,2), ...
                data.predictions(k,3),data.predictions(k,4),data.predictions(k,5)};
        end
        examples=[examples;struct('seed',data.seed,'dims',data.dims, ...
            'occupied_cells',nnz(~isnan(data.empty_grid.Y)), ...
            'empty_cells',nnz(isnan(data.empty_grid.Y)), ...
            'canonical_rank',data.canonical_rank,'test_rmse',data.test_rmse, ...
            'max_truncation_change',max(abs(data.imputed_grid.Y-data.truncated_tensor(:))))]; %#ok<AGROW>
    end
    maps=cell2table(map_rows,'VariableNames', ...
        {'dataset','index','x1','x2','reference','is_test','TS_G','TS_I','TS_T','Scattered','B'});
    writetable(maps,fullfile(output_dir,'snapshot_values.csv'));
    checks.visual_examples=examples;
    checks.changed_canonical_rank_at20pct=sum(rank_changes,1);
    fid=fopen(fullfile(output_dir,'verification.json'),'w','n','UTF-8');
    assert(fid>=0); cleanup=onCleanup(@() fclose(fid));
    fprintf(fid,'%s\n',jsonencode(checks,'PrettyPrint',true));
    fprintf('Four figures and one table exported; 250 summaries checked. CNO max %.3g.\n',max_cno_residual);
end

function intermediate_figure(data,figure_dir,source_dir)
% INTERMEDIATE_FIGURE Export the rasterization-to-prediction diagnostic panels.
%   data is a saved Franke snapshot; figure_dir/source_dir receive PDF/PNG
%   exports. Creates and closes the figure; returns no output.
    fig=new_figure([18.5,13.5]);
    layout=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
    limits=[min([data.training.Y;data.imputed_grid.Y;data.truncated_tensor(:)]), ...
        max([data.training.Y;data.imputed_grid.Y;data.truncated_tensor(:)])];
    ax=nexttile(layout);
    scatter(ax,data.training.X(:,1),data.training.X(:,2),14,data.training.Y,'filled');
    axis(ax,[0,1,0,1]); clim(ax,limits); colormap(ax,parula(128));
    title(ax,'(a) Training observations','FontWeight','normal');
    xlabel(ax,'x_1'); ylabel(ax,'x_2'); style_axes(ax,9);
    ax=nexttile(layout);
    draw_map(ax,data.empty_grid.X,data.empty_grid.Y,limits,parula(128));
    title(ax,sprintf('(b) Rasterized %d x %d',data.dims),'FontWeight','normal');
    xlabel(ax,'x_1'); ylabel(ax,'x_2');
    ax=nexttile(layout);
    draw_map(ax,data.imputed_grid.X,data.imputed_grid.Y,limits,parula(128));
    title(ax,'(c) TS-I imputation','FontWeight','normal');
    xlabel(ax,'x_1'); ylabel(ax,'x_2');
    ax=nexttile(layout);
    draw_map(ax,data.imputed_grid.X,data.truncated_tensor(:),limits,parula(128));
    title(ax,'(d) Truncated tensor','FontWeight','normal');
    xlabel(ax,'x_1'); ylabel(ax,'x_2');
    ax=nexttile(layout);
    spectrum=data.singular_values/data.singular_values(1);
    semilogy(ax,1:numel(spectrum),max(spectrum,eps),'-o', ...
        'Color',[0,.45,.7],'MarkerSize',4,'LineWidth',1.1); hold(ax,'on');
    yline(ax,.01,'--','Color',[.8,.2,.35]);
    title(ax,sprintf('(e) Singular spectrum\nrank %d',data.canonical_rank(1)), ...
        'FontWeight','normal','FontSize',9);
    xlabel(ax,'Index'); ylabel(ax,'Relative singular value'); grid(ax,'on'); style_axes(ax,9);
    ax=nexttile(layout);
    draw_map(ax,data.dataset.X,data.predictions(:,2),limits,parula(128));
    title(ax,sprintf('(f) Prediction\nRMSE %.4f',data.test_rmse(2)), ...
        'FontWeight','normal','FontSize',9);
    xlabel(ax,'x_1'); ylabel(ax,'x_2');
    cb=colorbar(ax); cb.Layout.Tile='east'; cb.Color='black'; cb.Label.String='Response';
    save_figure(fig,'r3_intermediate_steps',figure_dir,source_dir);
end

function comparison_figure(data,case_label,name,figure_dir,source_dir)
% COMPARISON_FIGURE Export reference, prediction, and test-error maps.
%   data is a snapshot; case_label selects axis labels; name is the file stem.
%   figure_dir/source_dir receive PDF/PNG exports. Returns no output.
    fig=new_figure([18.5,11.8]);
    ordering=[4,1,2,3]; % Scattered, TS-G, TS-I, TS-T.
    predictions=data.predictions(:,ordering);
    values=[data.dataset.Y;predictions(:)];
    response_limits=[min(values),max(values)];
    errors=predictions-data.dataset.Y;
    is_test=false(size(data.dataset.Y)); is_test(data.test_indices)=true;
    errors(~is_test,:)=NaN;
    error_limit=max(abs(errors),[],'all','omitnan');
    diverging=diverging_map;
    for row=1:2
        for column=1:5
            ax=axes(fig,'Position',[.065+(column-1)*.17, .60-(row-1)*.45, .14,.28]);
            if row==1
                if column==1
                    draw_map(ax,data.dataset.X,data.dataset.Y,response_limits,parula(128));
                    title_text='Reference';
                else
                    index=ordering(column-1);
                    draw_map(ax,data.dataset.X,data.predictions(:,index),response_limits,parula(128));
                    title_text=sprintf('%s\nRMSE %.4f',data.methods(index),data.test_rmse(index));
                end
            elseif column==1
                plot(ax,data.test.X(:,1),data.test.X(:,2),'.','Color',[.65,.65,.65],'MarkerSize',4);
                hold(ax,'on');
                plot(ax,data.training.X(:,1),data.training.X(:,2),'+','Color','black','MarkerSize',3);
                xlim(ax,[min(data.dataset.X(:,1)),max(data.dataset.X(:,1))]);
                ylim(ax,[min(data.dataset.X(:,2)),max(data.dataset.X(:,2))]);
                title_text='Train (+), test (.)'; style_axes(ax,9);
            else
                draw_map(ax,data.dataset.X,errors(:,column-1),[-error_limit,error_limit],diverging);
                title_text=sprintf('%s error',data.methods(ordering(column-1)));
            end
            letter=char('a'+(row-1)*5+column-1);
            title(ax,sprintf('(%c) %s',letter,title_text),'FontWeight','normal','FontSize',9);
            if contains(case_label,'NOAA')
                xlabel(ax,'Lon'); ylabel(ax,'Lat'); ax.XTick=[0,180,350]; ax.YTick=[-80,0,80];
            else
                xlabel(ax,'x_1'); ylabel(ax,'x_2'); ax.XTick=[0,.5,1]; ax.YTick=[0,.5,1];
            end
            if column>1
                ylabel(ax,''); ax.YTickLabel=[];
            end
            if column==5
                cb=colorbar(ax,'Position',[.93,.60-(row-1)*.45,.012,.28]);
                cb.Color='black'; cb.FontSize=8;
                if row==1, cb.Label.String='Response'; else, cb.Label.String='Prediction - reference'; end
            end
        end
    end
    save_figure(fig,name,figure_dir,source_dir);
end

function draw_map(ax,coordinates,values,limits,map)
% DRAW_MAP Render a Cartesian field with transparent missing values.
%   ax is the target axes; coordinates is [N x 2]; values has N entries.
%   limits is the color range and map an RGB colormap. Returns no output.
    [x,~,ix]=unique(coordinates(:,1)); [y,~,iy]=unique(coordinates(:,2));
    assert(numel(x)*numel(y)==size(coordinates,1));
    positions=sub2ind([numel(y),numel(x)],iy,ix);
    assert(numel(unique(positions))==numel(values));
    field=nan(numel(y),numel(x)); field(positions)=values;
    im=imagesc(ax,x,y,field); im.AlphaData=~isnan(field);
    assert(isequaln(field(positions),values));
    ax.YDir='normal'; clim(ax,limits); colormap(ax,map); style_axes(ax,9);
    ax.Color=[.88,.88,.88];
    xlim(ax,[min(x),max(x)]); ylim(ax,[min(y),max(y)]);
end

function write_noise_table(summary,tags,labels,path)
% WRITE_NOISE_TABLE Write the 20%-noise comparison as a LaTeX table.
%   summary contains noise statistics; tags/labels identify cases; path is
%   the destination file. Overwrites that file and returns no output.
    fid=fopen(path,'w','n','UTF-8'); assert(fid>=0);
    cleanup=onCleanup(@() fclose(fid));
    lines=["% Generated by experiments/export_reviewer3_materials.m."; ...
        "\begin{table}[pos=!htbp]"; "\centering"; ...
        "\caption{Sensibilidade a ruído adicional com desvio padrão $0.20s_{Y,\mathrm{train}}$, sob amostragem aleatória com $\rho=30\%$.}"; ...
        "\label{tab:r3-noise}"; "\small\begingroup"; ...
        "\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.15}"; ...
        "\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}ccccccc@{}}"; ...
        "\toprule"; ...
        "Conjunto & \shortstack{TS-I\\$\eta=0$} & \shortstack{TS-I\\$\eta=0.20$} & $D_{\mathrm{TS-I}}$ & $D_B$ & $D_{\mathrm{Scat.}}$ & $D_{\mathrm{TS-G}}$ & $D_{\mathrm{TS-T}}$ \\"; ...
        "\midrule"];
    fprintf(fid,'%s\n',lines);
    for d=1:10
        zero=summary(summary.dataset==tags(d) & summary.method=="TS-I" & summary.noise_fraction==0,:);
        high=summary(summary.dataset==tags(d) & summary.method=="TS-I" & summary.noise_fraction==.2,:);
        columns=["TS-I","B","Scattered","TS-G","TS-T"];
        deltas=zeros(1,5);
        for m=1:5
            entry=summary(summary.dataset==tags(d) & summary.method==columns(m) & ...
                summary.noise_fraction==.2,:);
            deltas(m)=entry.relative_mean_delta_pct;
        end
        line=sprintf('%s & \\rmsecell{%.4f}{%.4f} & \\rmsecell{%.4f}{%.4f}', ...
            labels(d),zero.mean_rmse,zero.sd_rmse,high.mean_rmse,high.sd_rmse);
        for m=1:5, line=[line,sprintf(' & $%+.2f$',deltas(m))]; end %#ok<AGROW>
        fprintf(fid,'%s %s\n',line,'\\');
    end
    tail=["\bottomrule\end{tabular*}\endgroup"; ...
        "\par\vspace{2pt}\begin{minipage}{\linewidth}\footnotesize"; ...
        "TS-I: média $\pm$ desvio padrão do RMSE entre 50 partições; unidades da função nos conjuntos sintéticos e $^\circ\mathrm{C}$ em NOAA. Para cada método, $D=100(\overline r_{0.20}-\overline r_0)/\overline r_0$ é a alteração percentual da média; valores positivos indicam piora. Os demais níveis e desvios padrão são preservados nos arquivos da análise. São resumos descritivos, sem teste de superioridade."; ...
        "\end{minipage}"; "\end{table}"];
    fprintf(fid,'%s\n',tail);
end

function map=diverging_map
% DIVERGING_MAP Return a 256-by-3 blue-white-orange RGB colormap.
%   Takes no arguments.
    blue=[.15,.35,.75]; white=[1,1,1]; orange=[.80,.25,.05];
    t=linspace(0,1,128)';
    map=[blue.*(1-t)+white.*t;white.*(1-t)+orange.*t];
end

function fig=new_figure(size_cm)
% NEW_FIGURE Create a hidden white figure with [width,height] size_cm.
%   fig is the figure handle; dimensions are in centimeters.
    fig=figure('Visible','off','Color','white','Units','centimeters', ...
        'Position',[1,1,size_cm]);
end

function style_axes(ax,font_size)
% STYLE_AXES Apply shared plot styling to axes handle ax at font_size points.
%   Updates graphics properties and returns no output.
    set(ax,'FontName','Arial','FontSize',font_size,'LineWidth',.6,'Box','on', ...
        'TickDir','out','Color','white','XColor','black','YColor','black','ZColor','black', ...
        'GridColor','black','GridAlpha',.12,'XMinorGrid','off','YMinorGrid','off');
    set([ax.Title,ax.XLabel,ax.YLabel,ax.ZLabel],'Color','black');
end

function save_figure(fig,name,figure_dir,source_dir)
% SAVE_FIGURE Export figure handle fig as PDF/PNG, then close it.
%   name is the file stem; figure_dir/source_dir receive PDF/PNG respectively.
%   Noise curves use vector PDF content; maps use raster content. No output.
    if strcmp(name,'r3_noise_sensitivity')
        exportgraphics(fig,fullfile(figure_dir,[name,'.pdf']),'ContentType','vector');
    else
        exportgraphics(fig,fullfile(figure_dir,[name,'.pdf']),'ContentType','image','Resolution',600);
    end
    exportgraphics(fig,fullfile(source_dir,[name,'.png']),'Resolution',180);
    close(fig);
end
