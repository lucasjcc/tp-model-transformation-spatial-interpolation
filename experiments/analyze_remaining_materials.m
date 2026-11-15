function analyze_remaining_materials
% ANALYZE_REMAINING_MATERIALS Compute paired tests and training-grid diagnostics.
%   Takes no arguments and returns no output. Run from the repository root
%   after analyze_current_ablation. Reuses paired errors without refitting,
%   reconstructs seeded grids, and checks their means against Monte Carlo CSVs.
%   Writes CSV/MAT summaries under results/reviewer4/cno;
%   Monte Carlo and ablation source records remain unchanged.

    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root,'auxiliary'));
    source = project_results_path('monte_carlo','cno','random_n50',true);
    ablation_dir = project_results_path('reviewer1','cno','',true);
    output = project_results_path('reviewer4','cno');
    if ~isfolder(output), mkdir(output); end
    names = ["franke","peaks","sinc","gauss_mixture",repmat("wind_temperature",1,6)];
    months = [1,1,1,1,1:6];
    ratios = [.1,.3,.5];
    rows = struct([]);
    grid_rows = struct([]);
    for d = 1:10
        tag = names(d);
        if d >= 5, tag = sprintf('wind_temperature_month%02d',months(d)); end
        data = get_dataset(20,names(d),months(d));
        for j = 1:3
            raw = load(fullfile(ablation_dir,sprintf('ablation_%s_ratio%d.mat',tag,round(100*ratios(j)))));
            assert(isequal(size(raw.rmse),[50,3]) && raw.record.csv_summary_matches);
            assert(isequal(raw.seeds,(1:50)'+50*(j-1)));
            delta = raw.rmse(:,2)-raw.rmse(:,3);
            se = std(delta)*sqrt(1/50+mean(raw.counts(:,2)./raw.counts(:,1)));
            if se == 0
                p = double(mean(delta)==0);
            else
                p = 2*tcdf(-abs(mean(delta)/se),49);
            end
            width = tinv(.975,49)*se;
            record = struct('dataset',string(tag),'ratio',ratios(j),'n',50, ...
                'mean_ts_i',mean(raw.rmse(:,2)),'sd_ts_i',std(raw.rmse(:,2)), ...
                'mean_scattered',mean(raw.rmse(:,3)),'sd_scattered',std(raw.rmse(:,3)), ...
                'mean_delta',mean(delta),'sd_delta',std(delta),'corrected_se',se,'p',p, ...
                'ci_low',mean(delta)-width,'ci_high',mean(delta)+width, ...
                'wins_ts_i',sum(delta<0),'csv_summary_matches',true, ...
                'max_prediction_difference_cno',raw.record.max_prediction_difference_cno);
            rows = [rows;record]; %#ok<AGROW>
            first = numel(grid_rows)+1;
            for k = 1:50
                rng(raw.seeds(k),'twister');
                [train,test] = split_dataset(data,ratios(j),names(d),'random');
                assert(isequal([numel(train.Y),numel(test.Y)],raw.counts(k,:)));
                [dims,info] = estimate_grid_size(train.X);
                [grid,stats] = create_grid_from_samples(train,dims,'none');
                observed = reshape(~isnan(grid.Y),dims);
                entry = struct('dataset',string(tag),'ratio',ratios(j),'seed',raw.seeds(k), ...
                    'n_train',numel(train.Y),'nx',dims(1),'ny',dims(2), ...
                    'cells',prod(dims),'empty_pct',stats.empty_cells_pct, ...
                    'requested_nx',info.I_est_k(1),'requested_ny',info.I_est_k(2), ...
                    'L_min',info.L_min,'L_max',info.L_max, ...
                    'at_lower',any(dims==info.L_min),'at_upper',any(dims==info.L_max), ...
                    'clipped_lower',any(info.lower_limit_active),'clipped_upper',any(info.upper_limit_active), ...
                    'empty_rows',sum(~any(observed,2)),'empty_columns',sum(~any(observed,1)));
                grid_rows = [grid_rows;entry]; %#ok<AGROW>
            end
            group = struct2table(grid_rows(first:end));
            published = readtable(fullfile(source,sprintf('errors_%s_ratio%d_random.csv',tag,round(100*ratios(j)))), ...
                'TextType','string');
            current = [mean(group.nx),mean(group.ny),mean(group.cells),mean(group.empty_pct)];
            for method = ["TS (Global Mean)","TS (Interp)","TS (Tensor Comp)"]
                r = published(published.Row==method,:);
                assert(height(r)==1 && r.N_grid==50);
                assert(max(abs(current-[r.MeanGridX1,r.MeanGridX2,r.MeanGridCells,r.MeanEmptyCellsPct]))<1e-10, ...
                    'remaining:GridMismatch','Grid replay differs from current CSV.');
            end
            fprintf('%s rho=%.1f: paired p=%.6g; 50 training-grid diagnostics match CSV.\n',tag,ratios(j),p);
        end
    end
    summary = struct2table(rows);
    [sorted_p,order] = sort(summary.p);
    adjusted = min(1,cummax((31-(1:30)').*sorted_p));
    summary.p_holm = nan(30,1);
    summary.p_holm(order) = adjusted;
    summary.reject_005 = summary.p_holm < .05;
    grids = struct2table(grid_rows);
    writetable(summary,fullfile(output,'paired_summary.csv'));
    writetable(grids,fullfile(output,'grid_observations.csv'));
    environment = struct('matlab_version',version,'toolboxes',ver,'completed_at',char(datetime('now')));
    save(fullfile(output,'remaining_analysis.mat'),'summary','grids','environment');
    fprintf('Validated 1,500 grids and paired TS-I/Scattered errors; %d Holm rejections.\n',sum(summary.reject_005));
end
