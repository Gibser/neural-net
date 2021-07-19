function net = net_conv_FC(layers, actv_functions, deriv_func, n_layers)
    %layers è un array di struct del tipo
    %{
        {
          type:      0 per livello FC, 1 per livello conv
          n_neurons: numero di neuroni se il livello è FC, numero di filtri
                     se è conv
          dim:     se type è conv, indica le dimensioni dei kernel
        }
    %}
    
    SIGMA = 0.1;
    net.weights = {};
    net.biases = {};
    net.activations = {};
    net.deriv_func = {};

    for i=2 : n_layers-1
       if layers{i}.type == 1   %Livello convoluzionale
           net.weights{i-1} = flatten_kernel(randn(layers{i}.dim(1), randn(layers{i}.dim(2))));
       elseif layers{i}.type == 0   %Livello full connected
           net.weights{i-1} = SIGMA*randn(layers{i}.n_neurons, layers{i-1}.n_neurons);
       end
       net.biases{i-1} = SIGMA*randn(n_neurons(i), 1);%gpuArray(SIGMA*randn(n_neurons(i), 1));
       net.activations{i-1} = actv_functions{i-1};
       net.deriv_func{i-1} = deriv_func{i-1};
    end
end
