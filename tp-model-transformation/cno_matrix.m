function [W, V, info] = cno_matrix(U, tol, options)
% CNO_MATRIX Close-to-normalized convex simplex decomposition U=W*V.
%   U: finite real [samples x components]; tol: reconstruction tolerance.
%   options fields: Starts (3), MaxIterations (200),
%   MaxFunctionEvaluations (2000), Exponent (1), EndpointWeight (5), Seed (0).
%   W contains convex weights; V maps them back to U. info reports affine
%   rank, reconstruction residual, conditioning, objective, and search status.
%   Uses TP Tool's cno/closeness/polarorto tangent-simplex construction and
%   logarithmic column-maximum objective, with bounded deterministic search.
%   Affine reduction handles constant columns. CNO is a heuristic search,
%   not a guarantee that every maximum equals one or a global optimum.
%   Copyright (c) 2006-2009, TP Tool authors; see THIRD_PARTY_NOTICES.md.
    if nargin < 2, tol = 1e-6; end
    if nargin < 3, options = struct(); end
    validateattributes(tol, {'numeric'}, {'scalar','real','finite','positive'});
    defaults = struct('Starts',3,'MaxIterations',200, ...
        'MaxFunctionEvaluations',2000,'Exponent',1,'EndpointWeight',5,'Seed',0);
    if ~isstruct(options) || ~isscalar(options)
        error('tp:BadCNOOptions','CNOOptions must be a scalar struct.');
    end
    names = fieldnames(options);
    for k = 1:numel(names)
        if ~isfield(defaults,names{k})
            error('tp:BadCNOOptions','Unknown CNO option: %s.',names{k});
        end
        defaults.(names{k}) = options.(names{k});
    end
    options = defaults;
    for name = {'Starts','MaxIterations','MaxFunctionEvaluations'}
        validateattributes(options.(name{1}),{'numeric'}, ...
            {'scalar','real','finite','integer','positive'});
    end
    validateattributes(options.Exponent,{'numeric'},{'scalar','real','finite','positive'});
    validateattributes(options.EndpointWeight,{'numeric'},{'scalar','real','finite','nonnegative'});
    validateattributes(options.Seed,{'numeric'}, ...
        {'scalar','real','finite','integer','>=',0,'<=',2^32-1});
    U = double(U);
    [Q, affine_info] = convex_affine_coordinates(U);
    d = size(Q,2);
    evaluations = 0;
    if d == 0
        W = ones(size(U,1),1);
        initial_objective = 0;
        best_objective = 0;
        exitflags = [];
    elseif d == 1
        q = Q(:,1);
        first = (q-min(q))/(max(q)-min(q));
        W = [first,1-first];
        initial_objective = 0;
        best_objective = 0;
        exitflags = [];
    else
        % An SNNN decomposition is a feasible incumbent. A failed local
        % search cannot replace it with a singular or nonconvex candidate.
        W = sn_nn_matrix(Q,tol);
        best_objective = closeness(W,options);
        initial_objective = best_objective;
        m = d+1;
        directions = null(ones(1,m));
        directions = directions./vecnorm(directions,2,2);
        initial_angles = to_angles(directions);
        stream = RandStream('mt19937ar','Seed',options.Seed);
        search_options = optimset('Display','off','MaxIter',options.MaxIterations, ...
            'MaxFunEvals',options.MaxFunctionEvaluations,'TolX',1e-6,'TolFun',1e-8);
        exitflags = zeros(1,options.Starts);
        for start = 1:options.Starts
            if start == 1
                angles = initial_angles;
            else
                angles = (rand(stream,m,d-1)-0.5)*pi;
                angles(:,end) = 2*angles(:,end);
            end
            % Retain feasible start points even if the optimizer exhausts
            % its budget before returning a better point.
            consider(angles);
            [angles,~,exitflags(start),output] = fminsearch(@objective,angles,search_options);
            evaluations = evaluations+output.funcCount;
            consider(angles);
        end
    end
    [V, info] = convex_vertex_map(W,U,tol);
    info.method = 'CNO';
    info.affine_rank = affine_info.affine_rank;
    info.degenerate_constant = d == 0;
    info.initial_objective = initial_objective;
    info.objective = best_objective;
    info.function_evaluations = evaluations;
    info.exitflags = exitflags;
    info.options = options;

    function value = objective(angles)
        % OBJECTIVE Score a simplex orientation using the enclosing Q/options.
        %   angles parameterizes face directions; value penalizes invalid weights.
        candidate_weights = tangent_simplex(Q,angles);
        if isempty(candidate_weights)
            value = 1e6;
        else
            value = closeness(candidate_weights,options);
        end
    end

    function consider(angles)
        % CONSIDER Update the enclosing best weights for a candidate orientation.
        %   angles parameterizes face directions; no output is returned.
        candidate_weights = tangent_simplex(Q,angles);
        if ~isempty(candidate_weights)
            value = closeness(candidate_weights,options);
            if value < best_objective
                W = candidate_weights;
                best_objective = value;
            end
        end
    end
end

function angles = to_angles(directions)
% TO_ANGLES Convert an [m x d] direction matrix to [m x (d-1)] angular coordinates.
%   Each input row specifies one simplex direction.
    [m,d] = size(directions);
    angles = zeros(m,d-1);
    for k = 1:d-2
        angles(:,k) = atan2(directions(:,k),vecnorm(directions(:,k+1:end),2,2));
    end
    angles(:,d-1) = atan2(directions(:,d-1),directions(:,d));
end

function W = tangent_simplex(Q,angles)
% TANGENT_SIMPLEX Construct convex weights for a candidate enclosing simplex.
%   Q is [n x d]; angles is [(d+1) x (d-1)]. W is [n x (d+1)], or empty
%   if the simplex is singular or fails the convexity checks.
    [n,d] = size(Q);
    m = d+1;
    vertices = zeros(m,m);
    for k = 1:d-1
        vertices(:,k) = sin(angles(:,k)).*prod(cos(angles(:,1:k-1)),2);
    end
    vertices(:,d) = prod(cos(angles),2);
    vertices(:,m) = 1-sum(vertices(:,1:d),2);
    P = [Q,1-sum(Q,2)];
    vertices = vertices+P(1,:)-mean(vertices,1);
    W = [];
    for k = 1:m
        if rcond(vertices) < 1e-12 || any(~isfinite(vertices(:))), return; end
        weights = P/vertices;
        minimum = min(weights(:,k));
        vertices = vertices-minimum*(vertices-vertices(k,:));
        vertices(:,m) = 1-sum(vertices(:,1:d),2);
    end
    if rcond(vertices) < 1e-12 || any(~isfinite(vertices(:))), return; end
    weights = P/vertices;
    if any(~isfinite(weights(:))) || min(weights(:)) < -1e-10 || ...
            norm(sum(weights,2)-1,inf) > 1e-10 || n < m
        return;
    end
    W = weights;
end

function value = closeness(W,options)
% CLOSENESS Score how closely convex-weight column maxima approach one.
%   W is a weight matrix; options supplies Exponent and EndpointWeight.
%   value is the logarithmic objective, with a penalty for nonpositive maxima.
    maxima = max(W,[],1);
    if any(maxima <= 0)
        value = 1e6;
        return;
    end
    value = norm(log(maxima),options.Exponent);
    endpoint_maxima = sort(max(W([1,end],:),[],1));
    if options.EndpointWeight > 0
        value = value+options.EndpointWeight* ...
            norm(log(endpoint_maxima(end-1:end)),options.Exponent);
    end
end
