function [y_pred, net, training_record] = neural_network_predict(Train_Dataset, Test_Dataset, varargin)
    % NEURAL_NETWORK_PREDICT Fits the manuscript's scalar-regression MLP.
    %
    % Inputs:
    %   Train_Dataset: finite real .X (N x Features) and .Y (N x 1).
    %                  Row-vector responses are also accepted.
    %   Test_Dataset:  finite real .X (M x Features); .Y is never used.
    %   varargin:      Optional Name-Value pairs:
    %                  - 'HiddenNeurons': Positive integer hidden-layer sizes
    %                    (default 20: one hidden layer as in the manuscript).
    %                  - 'TrainFcn': 'trainbr' (default), 'trainlm', or 'trainscg'.
    %                  - 'ShowWindow': true/false to display training GUI
    %                  - 'MaxEpochs': Positive integer epoch limit (default 1000).
    %
    % Output:
    %   y_pred: Vector (M x 1), in the original response units.
    %   net: Trained network, including preprocessing fitted to training data.
    %   training_record: MATLAB training diagnostics and internal split indices.
    %
    % Hidden activations are tansig; the scalar output is purelin. Training
    % inputs/targets use removeconstantrows and mapminmax. For trainbr, all
    % provided training samples are used, without internal validation/test.
    % The other algorithms use an internal 80/20 training/validation split.
    % Initialization uses the caller's RNG state; set rng before calling for
    % reproducibility. This function does not reset the RNG or use test data
    % to configure preprocessing, initialize weights, or select a model.

    p = inputParser;
    p.PartialMatching = false;
    addParameter(p, 'HiddenNeurons', 20, @is_hidden_sizes);
    addParameter(p, 'TrainFcn', 'trainbr', @is_training_function);
    addParameter(p, 'ShowWindow', false, @(value) isscalar(value) && ...
        (islogical(value) || (isnumeric(value) && isreal(value) && ...
        isfinite(value) && ismember(value, [0, 1]))));
    addParameter(p, 'MaxEpochs', 1000, @(value) isnumeric(value) && ...
        isscalar(value) && isreal(value) && isfinite(value) && ...
        value >= 1 && value == fix(value));
    parse(p, varargin{:});
    opts = p.Results;

    if ~isstruct(Train_Dataset) || ~isscalar(Train_Dataset) || ...
            ~all(isfield(Train_Dataset, {'X', 'Y'})) || ...
            ~isstruct(Test_Dataset) || ~isscalar(Test_Dataset) || ...
            ~isfield(Test_Dataset, 'X')
        error('neural_network_predict:BadDataset', ...
            'Training data require X and Y; test data require X.');
    end
    X_train = Train_Dataset.X;
    Y_train = Train_Dataset.Y;
    X_test = Test_Dataset.X;
    validateattributes(X_train, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'}, ...
        mfilename, 'Train_Dataset.X');
    validateattributes(Y_train, {'numeric'}, {'vector', 'real', 'finite', ...
        'numel', size(X_train, 1)}, mfilename, 'Train_Dataset.Y');
    validateattributes(X_test, {'numeric'}, {'2d', 'real', 'finite', ...
        'ncols', size(X_train, 2)}, mfilename, 'Test_Dataset.X');

    % MATLAB networks expect [features x samples] inputs and targets.
    
    x_train = double(X_train)';
    t_train = double(Y_train(:))';
    x_test = double(X_test)';

    training_function = lower(char(opts.TrainFcn));
    hidden_sizes = reshape(double(opts.HiddenNeurons), 1, []);
    net = fitnet(hidden_sizes, training_function);
    for layer = 1:numel(hidden_sizes)
        net.layers{layer}.transferFcn = 'tansig';
    end
    net.layers{end}.transferFcn = 'purelin';
    net.performFcn = 'mse';
    net.inputs{1}.processFcns = {'removeconstantrows', 'mapminmax'};
    net.outputs{end}.processFcns = {'removeconstantrows', 'mapminmax'};

    net.trainParam.showWindow = logical(opts.ShowWindow);
    net.trainParam.showCommandLine = false;
    net.trainParam.epochs = double(opts.MaxEpochs);
    
    % Bayesian regularization uses all supplied training samples. Only the
    % alternative trainers reserve an internal validation subset. External
    % test samples never participate in either internal subset.
    net.divideMode = 'sample';
    if strcmp(training_function, 'trainbr')
        net.divideFcn = 'dividetrain';
    else
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = 0.8;
        net.divideParam.valRatio = 0.2;
        net.divideParam.testRatio = 0;
    end

    [net, training_record] = train(net, x_train, t_train);

    if isempty(X_test)
        y_pred = zeros(0, 1);
        return;
    end
    y_pred_raw = net(x_test);
    
    y_pred = y_pred_raw(:);
    if any(~isfinite(y_pred))
        error('neural_network_predict:InvalidPrediction', ...
            'The trained network produced nonfinite predictions.');
    end
end

function valid = is_hidden_sizes(value)
% IS_HIDDEN_SIZES Validate a nonempty vector of positive integer layer sizes.
%   value is the candidate configuration; valid is a scalar logical.
    valid = isnumeric(value) && isvector(value) && ~isempty(value) && ...
        isreal(value) && all(isfinite(value)) && all(value >= 1 & value == fix(value));
end

function valid = is_training_function(value)
% IS_TRAINING_FUNCTION Check scalar text value against supported trainers.
%   valid is true for trainbr, trainlm, or trainscg (case-insensitive).
    valid = ((ischar(value) && isrow(value)) || ...
        (isstring(value) && isscalar(value))) && ...
        any(strcmpi(value, {'trainbr', 'trainlm', 'trainscg'}));
end
