function net=net(dim, n_hidden, n_output, n_layers)
    SIGMA = 0.1;
    net.weights = {};
    net.biases = {};
    net.activations = {};
    for i=1 : n_layers-1
        net.weights{i} = SIGMA*randn(n_hidden, dim);
        net.biases{i} = SIGMA*randn(n_hidden, 1);
        net.activations{i} = @sigmoid;
    end
    %livello output
    net.weights{end+1} = SIGMA*randn(n_output, n_hidden); 
    net.biases{end+1} = SIGMA*randn(n_output, 1);
    net.activations{end+1} = @identity;
    net.n_hidden=n_hidden;
    net.dim=dim;
    net.n_output=n_output;
    net.n_layers = n_layers;

end