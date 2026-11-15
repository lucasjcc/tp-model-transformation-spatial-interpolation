function directory = project_results_path(category, convexification, run_tag, allow_archive)
% PROJECT_RESULTS_PATH Locate outputs or read-only archived reference results.
%   category names a results group; convexification selects its model folder.
%   run_tag optionally identifies sampling and repetition count, e.g. random_n10.
%   allow_archive (default false) permits reading the previous folder layout
%   only when the new reference records are absent. Never enable it for outputs.
%   directory is an absolute path; this function does not create directories.
    if nargin < 3, run_tag = ''; end
    if nargin < 4, allow_archive = false; end
    category = validatestring(category, ...
        {'monte_carlo','reviewer1','reviewer2','reviewer3','reviewer4'});
    validateattributes(allow_archive, {'logical'}, {'scalar'});
    model = lower(strrep(char(convexification), '-', '_'));
    root = fileparts(fileparts(mfilename('fullpath')));
    directory = fullfile(root, 'results', category, model, char(run_tag));
    if allow_archive && ~has_reference_records(directory,category,run_tag)
        parents = dir(fullfile(root,'results',category));
        parents = parents([parents.isdir]);
        candidates = {};
        for k = 1:numel(parents)
            if any(strcmp(parents(k).name,{'.','..'})), continue; end
            archive = fullfile(parents(k).folder,parents(k).name,model);
            if has_reference_records(archive,category,run_tag)
                candidates{end+1} = archive; %#ok<AGROW>
            end
        end
        if numel(candidates) > 1
            error('results:AmbiguousArchive','Multiple archived reference folders were found.');
        elseif numel(candidates) == 1
            directory = candidates{1};
        end
    end
end

function available = has_reference_records(directory,category,run_tag)
% HAS_REFERENCE_RECORDS Check for the records needed by a results reader.
%   directory is a candidate folder; category/run_tag identify the records.
%   available is a scalar logical.
    if ~isfolder(directory)
        available = false;
        return;
    end
    switch category
        case 'monte_carlo'
            sampling = regexp(char(run_tag),'^(random|grid)_n\d+$','tokens','once');
            pattern = 'errors_*.csv';
            if ~isempty(sampling), pattern = ['errors_*_',sampling{1},'.csv']; end
            available = ~isempty(dir(fullfile(directory,pattern)));
        case 'reviewer1'
            available = isfile(fullfile(directory,'ablation_summary.csv'));
        case 'reviewer3'
            available = isfile(fullfile(directory,'noise_summary.csv'));
        case 'reviewer4'
            available = isfile(fullfile(directory,'paired_summary.csv'));
        otherwise
            available = true;
    end
end
