function Pm = n_mode(Pt,n)
% N_MODE Unfold tensor Pt into its mode-n matrix Pm.
%   n is the positive mode index; modes beyond ndims(Pt) are singleton.
%   Pm has size(Pt,n) rows and one column per combination of other indices.
siz = size(Pt);
N = length(siz);
if n> N
    siz = [siz ones(1,n-N)];
    N = n;
end
indice = cell(1,N);
Pm = zeros(siz(n),numel(Pt)/siz(n));
for ind = 1:numel(Pt)
    [indice{:}] = ind2sub(siz,ind);
    row = indice{n};
    if (n-1) > 0
        mul = siz(n-1);
        col = indice{n-1};
        for k = (n-2):(-1):1
            col = col + mul*(indice{k}-1);
            mul = mul*siz(k);
        end
        for k = N:(-1):(n+1)
            col = col + mul*(indice{k}-1);
            mul = mul*siz(k);
        end
    else
        mul = siz(N);
        col = indice{N};
        for k = (N-1):(-1):(n+1)
            col = col + mul*(indice{k}-1);
            mul = mul*siz(k);
        end
    end
    Pm(row,col) = Pt(ind);
end
end
