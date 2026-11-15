function Dataset = get_dataset(n_samples_per_input, chosen_dataset, chosen_mounth)
% GET_DATASET Generate a synthetic grid or load a monthly NOAA reference field.
%   n_samples_per_input is the node count per axis; [] retains native NOAA
%   resolution. chosen_dataset selects a supported benchmark name.
%   chosen_mounth is the NOAA month index (1..12), ignored for synthetic data.
%   Dataset has [N x 2] coordinates X and [N x 1] responses Y; NOAA rows with
%   missing responses are removed. Run from the repository root.
    avaliable_datasets = [
        "linear_sine" 
        "exp_2d"
        "franke"
        "peaks"
        "wind_temperature"
        "sinc"
        "gauss_mixture"
        "saddle"
    ];

    if ~ismember(chosen_dataset, avaliable_datasets)
        error("Invalid benchmark function. Options: %s", ...
        strjoin(avaliable_datasets, ", "));
    end

    if strcmpi(chosen_dataset, "wind_temperature")
        ncfile = './datasets/air.mon.v6.ltm.1991.2020.nc';
        
        if ~isfile(ncfile)
            error('NetCDF file not found: %s', ncfile);
        end
        
        lat = ncread(ncfile, 'lat');
        lon = ncread(ncfile, 'lon');
        
        % NOAA stores air as [longitude x latitude x month].
        raw_data = ncread(ncfile, 'air'); 
        
        temp_slice = raw_data(:, :, chosen_mounth);
        
        % Treat sentinel values below the physical range as missing.
        temp_slice(temp_slice < -1000) = NaN;

        [Lon, Lat] = meshgrid(lon, lat);
        
        % Match meshgrid's [latitude x longitude] layout.
        temp_slice = temp_slice'; 
        
        % Resample the reference field when a node count is supplied.
        if ~isempty(n_samples_per_input)
            new_lon = linspace(min(lon), max(lon), n_samples_per_input);
            new_lat = linspace(min(lat), max(lat), n_samples_per_input);
            [NewLon, NewLat] = meshgrid(new_lon, new_lat);
            
            temp_slice = interp2(Lon, Lat, temp_slice, NewLon, NewLat, 'linear');
            
            X1 = NewLon;
            X2 = NewLat;
        else
            X1 = Lon;
            X2 = Lat;
        end
        
        Dataset.X = [X1(:), X2(:)];
        Dataset.Y = temp_slice(:);
        
        % Exclude missing responses before passing data to predictors.
        valid_mask = ~isnan(Dataset.Y);
        Dataset.X = Dataset.X(valid_mask, :);
        Dataset.X = double(Dataset.X);
        Dataset.Y = Dataset.Y(valid_mask);
        Dataset.Y = double(Dataset.Y);
        return; 
    end

    if strcmpi(chosen_dataset, "linear_sine")
        x1 = linspace(0, 10, n_samples_per_input);
        x2 = linspace(-pi, pi, n_samples_per_input);

        [X1, X2] = meshgrid(x1, x2);

        Dataset.X = [X1(:), X2(:)];
        Y = linear_sine(X1, X2);
        Dataset.Y = Y(:);
        return;
    end

    if strcmpi(chosen_dataset, "exp_2d")
        x1 = linspace(0, 2, n_samples_per_input);
        x2 = linspace(0, 4, n_samples_per_input);

        [X1, X2] = meshgrid(x1, x2);

        Dataset.X = [X1(:), X2(:)];
        Y = exp_2d(X1, X2);
        Dataset.Y = Y(:);
        return;
    end

    if strcmpi(chosen_dataset, "franke")
        x1 = linspace(0, 1, n_samples_per_input);
        x2 = linspace(0, 1, n_samples_per_input);

        [X1, X2] = meshgrid(x1, x2);

        Dataset.X = [X1(:), X2(:)];
        Y = franke_func(X1, X2);
        Dataset.Y = Y(:);
        return;
    end

    if strcmpi(chosen_dataset, "peaks")
        x1 = linspace(-3, 3, n_samples_per_input);
        x2 = linspace(-3, 3, n_samples_per_input);

        [X1, X2] = meshgrid(x1, x2);

        Dataset.X = [X1(:), X2(:)];
        Y = peaks_func(X1, X2);
        Dataset.Y = Y(:);
        return;
    end

    if strcmpi(chosen_dataset, "sinc")
        x1 = linspace(-8, 8, n_samples_per_input);
        x2 = linspace(-8, 8, n_samples_per_input);
        [X1, X2] = meshgrid(x1, x2);
        Dataset.X = [X1(:), X2(:)];
        Y = sinc_2d(X1, X2);
        Dataset.Y = Y(:);
        return;
    end

    if strcmpi(chosen_dataset, "gauss_mixture")
        x1 = linspace(0, 1, n_samples_per_input);
        x2 = linspace(0, 1, n_samples_per_input);
        [X1, X2] = meshgrid(x1, x2);
        Dataset.X = [X1(:), X2(:)];
        Y = gauss_mixture(X1, X2);
        Dataset.Y = Y(:);
        return;
    end

    if strcmpi(chosen_dataset, "saddle")
        x1 = linspace(-3, 3, n_samples_per_input);
        x2 = linspace(-3, 3, n_samples_per_input);
        [X1, X2] = meshgrid(x1, x2);
        Dataset.X = [X1(:), X2(:)];
        Y = saddle_func(X1, X2);
        Dataset.Y = Y(:);
        return;
    end
end

function F = linear_sine(x, y)
% LINEAR_SINE Evaluate 2*x + sin(3*y) on equally sized coordinate arrays.
%   F has the same size as x and y.
    a = 2.0;
    b = 3.0;

    F = a*x + sin(b*y);
end

function F = exp_2d(x1, x2)
% EXP_2D Evaluate a shifted exponential benchmark on matching coordinate arrays.
%   F has the same size as x1 and x2.
    F = 10*(x1 .* exp(-x1.^2 - x2.^2)) + 27;
end

function F = franke_func(x, y)
% FRANKE_FUNC Evaluate Franke's surface on matching coordinate arrays x and y.
%   F is the corresponding response array, conventionally on [0,1]^2.
    term1 = 0.75 * exp(-((9*x - 2).^2 + (9*y - 2).^2) / 4);
    term2 = 0.75 * exp(-((9*x + 1).^2) / 49 - (9*y + 1) / 10);
    term3 = 0.5  * exp(-((9*x - 7).^2 + (9*y - 3).^2) / 4);
    term4 = -0.2 * exp(-((9*x - 4).^2 + (9*y - 7).^2));
    
    F = term1 + term2 + term3 + term4;
end

function F = peaks_func(x, y)
% PEAKS_FUNC Evaluate the peaks surface on matching coordinate arrays x and y.
%   F is the response array, conventionally evaluated on [-3,3]^2.
    F = 3*(1-x).^2.*exp(-(x.^2) - (y+1).^2) ...
      - 10*(x/5 - x.^3 - y.^5).*exp(-x.^2-y.^2) ...
      - 1/3*exp(-(x+1).^2 - y.^2);
end

function F = sinc_2d(x, y)
% SINC_2D Evaluate sin(r)/r for matching arrays x and y, r = hypot(x,y).
%   F has the input size and equals one at the origin.
    r = sqrt(x.^2 + y.^2);
    F = ones(size(r));
    nz = r ~= 0;
    F(nz) = sin(r(nz)) ./ r(nz);
end

function F = gauss_mixture(x, y)
% GAUSS_MIXTURE Evaluate four Gaussian peaks on matching arrays x and y.
%   F has the input size; component rows specify [center_x, center_y, amplitude, sigma].
    comps = [
        0.2, 0.2,  1.0, 0.08;
        0.7, 0.3,  0.8, 0.05;
        0.4, 0.75, 1.2, 0.06;
        0.8, 0.8,  0.6, 0.04
    ];
    F = zeros(size(x));
    for i = 1:size(comps,1)
        cx = comps(i,1); cy = comps(i,2);
        a  = comps(i,3); s  = comps(i,4);
        F = F + a * exp(- ((x - cx).^2 + (y - cy).^2) / (2*s^2));
    end
end

function F = saddle_func(x, y)
% SADDLE_FUNC Evaluate 1.5*(x.^2-y.^2) on matching coordinate arrays.
%   F is the saddle-shaped response array with the same size as x and y.
    F = 1.5 * (x.^2 - y.^2);
end
