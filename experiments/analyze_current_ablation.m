function summary = analyze_current_ablation(case_indices)
% ANALYZE_CURRENT_ABLATION Compare pre-truncation control B with truncated TP.
%   case_indices selects cases 1..10 (default all) in the Random-sampling
%   protocol. Run from the repository root. summary is a table of paired RMSE
%   statistics and validation checks; a full run adds Holm-adjusted tests.
%   Saves raw records under results/reviewer1/cno;
%   only a full run writes ablation_summary.csv. Published Monte Carlo CSVs
%   are read only; the replay does not recover overwritten individual records.

    if nargin < 1
        case_indices = 1:10;
    end
    validateattributes(case_indices, {'numeric'}, {'vector','integer','>=',1,'<=',10});
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root,'auxiliary'), fullfile(root,'interpolation-algorithm'), ...
        fullfile(root,'tp-model-transformation'));
    source_dir = project_results_path('monte_carlo','cno','random_n50',true);
    output_dir = project_results_path('reviewer1','cno');
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    names = ["franke","peaks","sinc","gauss_mixture",repmat("wind_temperature",1,6)];
    months = [1,1,1,1,1:6];
    ratios = [.1,.3,.5];
    n = 50;
    rows = struct([]);
    for case_index = case_indices
        name = names(case_index);
        dataset = get_dataset(20,name,months(case_index));
        options = spatial_method_options(name);
        tag = char(name);
        if name == "wind_temperature"
            tag = sprintf('%s_month%02d',tag,months(case_index));
        end
        for ratio_index = 1:3
            ratio = ratios(ratio_index);
            csv_path = fullfile(source_dir,sprintf('errors_%s_ratio%d_random.csv',tag,round(100*ratio)));
            published = readtable(csv_path,'TextType','string');
            seeds = (1:n)' + n*(ratio_index-1);
            rmse = nan(n,3); % B, truncated TP, direct Scattered.
            counts = nan(n,2);
            no_truncation_error = zeros(n,1);
            no_truncation_relative_error = zeros(n,1);
            cno_error = NaN;
            for iteration = 1:n
                rng(seeds(iteration));
                [training,test] = split_dataset(dataset,ratio,name,"random");
                counts(iteration,:) = [numel(training.Y),numel(test.Y)];
                dims = estimate_grid_size(training.X);
                grid = create_grid_from_samples(training,dims,'interp');
                tensor = reshape(grid.Y,dims);
                axes = {unique(grid.X(:,1)),unique(grid.X(:,2))};
                query = max([axes{1}(1),axes{2}(1)], ...
                    min([axes{1}(end),axes{2}(end)],test.X));
                before = griddedInterpolant(axes,tensor,'linear','none');
                prediction_before = before(query(:,1),query(:,2));
                [factors,core] = hosvd_sampled_func(tensor,0,1e-2);
                truncated = n_mode_product(n_mode_product(core,factors{1},1),factors{2},2);
                after = griddedInterpolant(axes,truncated,'linear','none');
                prediction_after = after(query(:,1),query(:,2));
                [left,~,right] = svd(tensor);
                full_tensor = left*(left'*tensor*right)*right';
                no_truncation_error(iteration) = max(abs(full_tensor-tensor),[],'all');
                no_truncation_relative_error(iteration) = ...
                    no_truncation_error(iteration)/max(1,max(abs(tensor),[],'all'));
                direct = interpolation_predict(training,test, ...
                    'Method',options.scattered_method,'Extrapolation','nearest');
                rmse(iteration,:) = [sqrt(mean((test.Y-prediction_before).^2)), ...
                    sqrt(mean((test.Y-prediction_after).^2)),sqrt(mean((test.Y-direct).^2))];
                if iteration == 1
                    [~,~,vertices,weights] = train_ts_model(grid,1e-2,1e-6, ...
                        'Convexification','CNO','CNOOptions',struct());
                    convex = estimate_data(test,grid,weights,vertices);
                    cno_error = max(abs(convex.Y-prediction_after));
                    assert(cno_error <= 1e-6*max(1,max(abs(prediction_after))), ...
                        'ablation:CNOFailure','CNO and canonical predictions disagree.');
                end
            end
            assert(all(isfinite(rmse),'all') && max(no_truncation_relative_error) < 1e-12);
            % Published rounding is the strongest available replay comparison:
            % individual Random records were replaced by the later Grid run.
            methods = ["TS (Interp)","Scattered"];
            current = [mean(rmse(:,[2,3]));std(rmse(:,[2,3]),0,1)];
            published_values = nan(2,2);
            for column = 1:2
                entry = published(published.Row == methods(column),:);
                assert(height(entry) == 1 && entry.N_ok == n);
                tokens = regexp(char(entry.RMSE),'^(\d+\.\d{3})\s*±\s*(\d+\.\d{3})$', ...
                    'tokens','once');
                assert(~isempty(tokens));
                published_values(:,column) = str2double(tokens(:));
            end
            csv_difference = max(abs(current-published_values),[],'all');
            assert(csv_difference <= 0.0005001, ...
                'ablation:CSVFailure','Replay does not match current Random CSV: %s',csv_path);
            delta = rmse(:,2)-rmse(:,1);
            corrected_se = std(delta,0)*sqrt(1/n+mean(counts(:,2)./counts(:,1)));
            if corrected_se == 0
                p = double(mean(delta)==0);
            else
                p = 2*tcdf(-abs(mean(delta)/corrected_se),n-1);
            end
            half_width = tinv(.975,n-1)*corrected_se;
            record = struct('dataset',string(tag),'ratio',ratio,'n',n, ...
                'mean_before',mean(rmse(:,1)),'sd_before',std(rmse(:,1)), ...
                'mean_tp',mean(rmse(:,2)),'sd_tp',std(rmse(:,2)), ...
                'mean_scattered',mean(rmse(:,3)),'sd_scattered',std(rmse(:,3)), ...
                'mean_delta',mean(delta),'sd_delta',std(delta),'corrected_se',corrected_se, ...
                'test_train_ratio',mean(counts(:,2)./counts(:,1)),'p',p, ...
                'ci_low',mean(delta)-half_width,'ci_high',mean(delta)+half_width, ...
                'max_csv_rounding_difference',csv_difference,'csv_summary_matches',true, ...
                'max_prediction_difference_cno',cno_error, ...
                'max_no_truncation_error',max(no_truncation_error));
            rows = [rows;record]; %#ok<AGROW>
            matlab_version = version;
            save(fullfile(output_dir,sprintf('ablation_%s_ratio%d.mat',tag,round(100*ratio))), ...
                'rmse','seeds','counts','options','record','no_truncation_error', ...
                'no_truncation_relative_error','matlab_version');
            fprintf('%s rho=%.1f: TP-B=%+.9g, p=%.5g, CSV match=1, CNO error=%.3g\n', ...
                tag,ratio,mean(delta),p,cno_error);
        end
    end
    summary = struct2table(rows);
    if numel(unique(case_indices)) == 10
        assert(height(summary) == 30);
        [ordered_p,order] = sort(summary.p);
        adjusted_p = min(1,cummax((31-(1:30)').*ordered_p));
        summary.p_holm = nan(30,1);
        summary.p_holm(order) = adjusted_p;
        summary.reject_005 = summary.p_holm < .05;
        writetable(summary,fullfile(output_dir,'ablation_summary.csv'));
        save(fullfile(output_dir,'summary.mat'),'summary');
        fprintf('Validated 1,500 replay pairs; %d Holm rejections in 30 contrasts.\n',sum(summary.reject_005));
    end
end
