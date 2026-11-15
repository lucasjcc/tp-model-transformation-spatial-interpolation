function P = n_mode_product(A,U,n)
% N_MODE_PRODUCT Multiply tensor A by matrix U along mode n.
%   U must have one column per element of A along mode n. P is the
%   resulting tensor, with size(U,1) elements along that mode.
    Pm = n_mode(A,n);
    Pm = U*Pm;
    size_vector = size(A);
    size_vector(n) = size(U,1);
    P = n_mode2tensor(Pm,n,size_vector);
end
