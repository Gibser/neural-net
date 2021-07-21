function net=net(n_neurons, actv_functions, deriv_func, n_layers)
    SIGMA = 0.1;
    net.weights = {};
    net.biases = {};
    net.weightMomentums = {};
    net.activations = {};
    net.biasMomentums = {};
    for i=2 : n_layers
        net.weights{i-1} = SIGMA*randn(n_neurons(i), n_neurons(i-1));
        net.biases{i-1} = SIGMA*randn(n_neurons(i), 1);
        net.weightMomentums{i-1} = zeros(size(net.weights{i-1})) + 0.1;         %DELTA per i pesi per l'RPROP
        net.biasMomentums{i-1} = SIGMA*randn(n_neurons(i), 1);
        net.activations{i-1} = actv_functions{i-1};
        net.deriv_func{i-1} = deriv_func{i-1};
    end
    net.n_neurons=n_neurons;
    net.n_layers = n_layers;
end