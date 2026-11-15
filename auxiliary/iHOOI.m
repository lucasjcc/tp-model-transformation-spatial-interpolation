function [U, S, X_filled, info] = iHOOI(Tensor_obs, singular_value_tolerance, varargin)
    % IHOOI Complete missing tensor entries using adaptive-rank HOOI.
    %   Tensor_obs is a numeric tensor with NaNs marking missing entries.
    %   singular_value_tolerance is a positive energy tolerance: rank grows
    %   until the observed relative residual is at most its square root or
    %   the rank cap is reached. This is not an HOSVD singular-value cutoff.
    %   Options: MaxIter (100), Tol (1e-6, missing-entry relative change),
    %   Init ('mean'; other values use zero), Verbose (false), RankStep (2),
    %   and MaxRankFrac (0.5 of the smallest tensor dimension).
    %   U contains mode factors; S is the projected core; X_filled preserves
    %   observations and fills missing cells. info reports ranks, iteration
    %   count, and the observed-entry reconstruction residual.

    p = inputParser;
    addParameter(p,'MaxIter', 100, @(x)isscalar(x) && x>0);
    addParameter(p,'Tol', 1e-6, @isnumeric);
    addParameter(p,'Init', 'mean', @(s)ischar(s)||isstring(s));
    addParameter(p,'Verbose', false, @islogical);
    addParameter(p,'RankStep', 2, @(x)isscalar(x) && x>=1);
    addParameter(p,'MaxRankFrac', 0.5, @(x)isscalar(x) && x>0 && x<=1);
    parse(p, varargin{:});
    opts = p.Results;

    Omega = ~isnan(Tensor_obs);
    if ~any(Omega(:)), error('Tensor is completely empty (all NaN).'); end

    sz = size(Tensor_obs);
    Nmode = ndims(Tensor_obs);

    X = Tensor_obs;
    obs_vals = Tensor_obs(Omega);

    if strcmpi(opts.Init, 'mean')
        fillval = mean(obs_vals, 'all');
    else
        fillval = 0;
    end
    X(~Omega) = fillval;

    obs_norm = max(norm(obs_vals), eps);
    target_relres = sqrt(singular_value_tolerance);
    max_rank = max(1, floor(opts.MaxRankFrac * min(sz)));

    ranks = ones(1, Nmode);
    U = cell(1, Nmode);
    total_iter = 0;

    % Grow ranks from one rather than estimating them from the initial fill.
    while true
        for n = 1:Nmode
            G = unfold(X, n);
            r_n = min(ranks(n), min(size(G)));
            [Un, ~, ~] = svds(double(G), r_n);
            U{n} = Un;
        end

        % Update factors and missing entries while holding observations fixed.
        for iter = 1:opts.MaxIter
            total_iter = total_iter + 1;
            for n = 1:Nmode
                Y = X;
                for m = 1:Nmode
                    if m == n, continue; end
                    Y = n_mode_product(Y, U{m}', m);
                end
                G_n = unfold(Y, n);
                r_n = min(ranks(n), min(size(G_n)));
                [Un, ~, ~] = svds(double(G_n), r_n);
                U{n} = Un;
            end

            S_core = X;
            for n = 1:Nmode
                S_core = n_mode_product(S_core, U{n}', n);
            end
            X_rec = S_core;
            for n = 1:Nmode
                X_rec = n_mode_product(X_rec, U{n}, n);
            end

            X_new = X;
            X_new(~Omega) = X_rec(~Omega);

            missing_prev = X(~Omega);
            missing_new  = X_new(~Omega);
            if isempty(missing_prev)
                relchg = 0;
            else
                relchg = norm(missing_new - missing_prev) / max(1e-12, norm(missing_prev));
            end

            X = X_new;

            if opts.Verbose
                fprintf('rank=%s iter %d: relchg(missing) = %.3e\n', mat2str(ranks), iter, relchg);
            end

            if relchg < opts.Tol
                break;
            end
        end

        % Evaluate the low-rank reconstruction, not the observation-preserving fill.
        relres_obs = norm(X_rec(Omega) - Tensor_obs(Omega)) / obs_norm;

        if opts.Verbose
            fprintf('[iHOOI] rank=%s: relres_on_observed=%.3e (target=%.3e)\n', mat2str(ranks), relres_obs, target_relres);
        end

        if relres_obs <= target_relres || all(ranks >= max_rank)
            break;
        end

        ranks = min(ranks + opts.RankStep, max_rank);
    end

    S = X;
    for n = 1:Nmode
        S = n_mode_product(S, U{n}', n);
    end

    X_filled = X;
    info.ranks = ranks;
    info.iter = total_iter;
    info.relres_observed = relres_obs;
end

function Xn = unfold(X, mode)
% UNFOLD Reshape tensor X into a matrix with mode as its row dimension.
%   mode is a valid tensor dimension; Xn contains all other indices as columns.
    sz = size(X);
    N = ndims(X);
    perm = [mode, 1:mode-1, mode+1:N];
    Xp = permute(X, perm);
    Xn = reshape(Xp, sz(mode), []);
end

function T = n_mode_product(T, M, mode)
% N_MODE_PRODUCT Multiply tensor T by matrix M along the selected mode.
%   M has size(T,mode) columns; output T has size(M,1) entries along that mode.
    sz = size(T);
    N = ndims(T);
    perm = [mode, 1:mode-1, mode+1:N];
    Tperm = permute(T, perm);
    Tmat = reshape(Tperm, sz(mode), []);
    Tmat = M * Tmat;
    new_sz = [size(M,1), sz(perm(2:end))];
    T = ipermute(reshape(Tmat, new_sz), perm);
end
