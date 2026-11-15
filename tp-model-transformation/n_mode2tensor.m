function Pt = n_mode2tensor(Pm,n,size_vector)
% N_MODE2TENSOR Reconstruct a tensor from its mode-n unfolding.
%   Pm is the unfolded matrix, n is its mode, and size_vector specifies
%   the original tensor dimensions. Pt is the reconstructed tensor.
N = length(size_vector);
indice = cell(1,N);
Pt = zeros(size_vector);
for ind = 1:numel(Pt)
    [indice{:}] = ind2sub(size_vector,ind);
    row = indice{n};
    if (n-1) > 0
        mul = size_vector(n-1);
        col = indice{n-1};
        for k = (n-2):(-1):1
            col = col + mul*(indice{k}-1);
            mul = mul*size_vector(k);
        end
        for k = N:(-1):(n+1)
            col = col + mul*(indice{k}-1);
            mul = mul*size_vector(k);
        end
    else
        mul = size_vector(N);
        col = indice{N};
        for k = (N-1):(-1):(n+1)
            col = col + mul*(indice{k}-1);
            mul = mul*size_vector(k);
        end
    end
    Pt(ind) = Pm(row,col);
end
end
