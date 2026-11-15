function [W, V, info] = convexify_matrix(U, type, tol, cno_options)
% CONVEXIFY_MATRIX Convex decomposition U=W*V following TP Tool genhull.
%   U is a nonempty, finite real [samples x components] factor matrix.
%   type: 'SNNN', 'CNO', 'RNO-INO' (case-insensitive).
%   tol: reconstruction/property tolerance (default 1e-6).
%   cno_options: optional scalar struct accepted by cno_matrix.
%   W: weights; V: vertex map to apply to the canonical core mode.
%   info: method, residual, conditioning and weight-property diagnostics.
    if nargin < 2, type = 'RNO-INO'; end
    if nargin < 3, tol = 1e-6; end
    if nargin < 4, cno_options = struct(); end
    choices = {'SNNN','CNO','RNO-INO'};
    if ~((ischar(type) && isrow(type)) || (isstring(type) && isscalar(type))) || ...
            ~any(strcmpi(type,choices))
        error('tp:UnknownConvexification','Choose SNNN, CNO, or RNO-INO.');
    end
    type = upper(char(type));
    switch type
        case 'SNNN'
            [W,~,info] = sn_nn_matrix(U,tol);
            V = info.vertices;
        case 'RNO-INO'
            initial = sn_nn_matrix(U,tol);
            [W,~,info] = rno_ino_matrix(initial,tol);
            [V, map_info] = convex_vertex_map(W,U,tol);
            info.reconstruction_error = map_info.reconstruction_error;
        case 'CNO'
            [W,V,info] = cno_matrix(U,tol,cno_options);
    end
end
