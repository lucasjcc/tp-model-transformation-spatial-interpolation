function [absolute_error, MAE, RMSE, RE, confidence, MaxAE, StdErr] = calculate_erros(real_data, estimated_data)
% CALCULATE_ERROS Compute prediction errors and a local confidence map.
%   real_data and estimated_data must have the same size. absolute_error
%   contains elementwise absolute errors; MAE, RMSE, RE, MaxAE and StdErr
%   are aggregate metrics. confidence is the local standard deviation of
%   absolute_error, computed when at least five outputs are requested.

    if ~isequal(size(real_data), size(estimated_data))
        error('Real and estimated data matrices must have the same size.');
    end

    signed_error = double(real_data) - double(estimated_data);

    absolute_error = abs(real_data - estimated_data);

    MAE = mean(absolute_error(:));
    RMSE = sqrt(mean(signed_error(:).^2));
    RE = norm(signed_error, 'fro') / norm(double(real_data), 'fro');
    MaxAE = max(absolute_error(:));
    StdErr = std(signed_error(:));

    if nargout >= 5
        confidence = stdfilt(absolute_error);
    end
end
