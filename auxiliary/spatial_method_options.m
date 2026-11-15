function options = spatial_method_options(dataset_name)
% SPATIAL_METHOD_OPTIONS Return dataset-specific predictor settings.
%   dataset_name is a benchmark name. options contains the variogram type,
%   fitting flag, and scattered interpolation method.
    options = struct('kriging_type','gaussian','kriging_fit',false, ...
        'scattered_method','natural');
    if strcmpi(dataset_name,'wind_temperature')
        options.kriging_type = 'exponential';
        options.kriging_fit = true;
        options.scattered_method = 'linear';
    end
end
