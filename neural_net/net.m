function net=net(n_neurons, actv_functions, n_layers)
    SIGMA = 0.1;
    net.weights = {};
    net.biases = {};
    net.activations = {};
    for i=2 : n_layers
        net.weights{i-1} = SIGMA*randn(n_neurons(i), n_neurons(i-1));
        net.biases{i-1} = SIGMA*randn(n_neurons(i), 1);
        net.activations{i-1} = actv_functions{i};
    end
    net.n_neurons=n_neurons;
    net.n_layers = n_layers;
end