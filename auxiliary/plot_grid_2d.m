function plot_grid_2d(Train_Dataset, Train_Dataset_Grid)
% PLOT_GRID_2D Overlay training coordinates and grid-node lines in a new figure.
%   Train_Dataset.X is [N x 2]; Train_Dataset_Grid.centers contains two axis
%   vectors. The function creates a figure and returns no output.
    
    centers_cell = Train_Dataset_Grid.centers;
    
    X_samples = Train_Dataset.X;

    
    edges_x = centers_cell{1};
    edges_y = centers_cell{2};

    
    figure;
    ax = gca;
    
    scatter(ax, X_samples(:,1), X_samples(:,2), 20, 'r', 'fill');
    
    hold(ax, 'on');
    box(ax, 'on');

    
    y_limits = [min(edges_y) max(edges_y)];
    x_limits = [min(edges_x) max(edges_x)];

    for i = 1:length(edges_x)
        plot(ax, [edges_x(i) edges_x(i)], y_limits, 'Color', [0.3 0.3 0.3], 'LineWidth', 0.5);
    end
    
    for j = 1:length(edges_y)
        plot(ax, x_limits, [edges_y(j) edges_y(j)], 'Color', [0.3 0.3 0.3], 'LineWidth', 0.5);
    end
    
    hold(ax, 'off');

    xlabel(ax, '$x_{1}$', 'Interpreter','latex');
    ylabel(ax, '$x_{2}$', 'Interpreter','latex');
    title(ax, 'Grid regular gerado');
    
    axis(ax, [min(edges_x) max(edges_x) min(edges_y) max(edges_y)]);
end
