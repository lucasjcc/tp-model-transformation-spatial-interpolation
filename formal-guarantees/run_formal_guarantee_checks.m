function report = run_formal_guarantee_checks(Dataset, Train_Dataset_Grid, Test_Dataset, S_rno, U_rno, T_sv)
% RUN_FORMAL_GUARANTEE_CHECKS Check convex TP structure and numerical bounds.
%   Dataset has dense reference X/Y values, including any prior resampling.
%   Train_Dataset_Grid has X, filled Y, and dims; Test_Dataset.X is [N x 2].
%   S_rno and U_rno are the convex core and mode factors; T_sv is the training
%   singular-value cutoff. Natural-neighbor reference interpolation supplies
%   a proxy field at grid nodes, not independently observed ground truth.
%   report aggregates model_validity, certified_bounds, robustness_modulus,
%   hosvd_truncation, and error_decomposition diagnostics.

    fprintf('\n================ FORMAL GUARANTEES VERIFICATION ================\n');

    report.model_validity = validate_convex_tp_model(U_rno, S_rno);

    report.certified_bounds = verify_certified_bounds(S_rno, U_rno, Train_Dataset_Grid, Test_Dataset);

    report.robustness_modulus = verify_robustness_modulus(S_rno, U_rno, Train_Dataset_Grid, Test_Dataset);

    dims = Train_Dataset_Grid.dims;
    Grid_Tensor = reshape(Train_Dataset_Grid.Y, dims);
    report.hosvd_truncation = hosvd_truncation_bound(Grid_Tensor, T_sv);

    % Proxy for the true field f at the grid nodes: natural-neighbor
    % interpolation of the dense reference Dataset
    F_true = scatteredInterpolant(Dataset.X(:,1), Dataset.X(:,2), Dataset.Y, 'natural', 'nearest');
    f_true_vec = F_true(Train_Dataset_Grid.X(:,1), Train_Dataset_Grid.X(:,2));
    report.error_decomposition = verify_error_decomposition(f_true_vec, Train_Dataset_Grid, S_rno, U_rno, T_sv);

    fprintf('==================================================================\n\n');
end
