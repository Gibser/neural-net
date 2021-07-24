function [x_new] = reshape_flatten(x, n_filters, feature_map_dim)
    n_imgs = size(x, 1) / feature_map_dim;
    x_new = zeros(feature_map_dim*n_filters, n_imgs);
    c = 1;
    for i = 1 : feature_map_dim : size(x, 1)-1
        tmp = x(i:i+feature_map_dim-1, 1:n_filters);
        x_new(:, c) = tmp(:);
        c = c + 1;
    end
end

