function net = net_conv_FC(layers, actv_functions, deriv_func, n_layers)
    %layers è un array di struct del tipo
    %{
        {
          type:      0 per livello FC, 1 per livello conv
          n_neurons: numero di neuroni se il livello è FC, numero di filtri
                     se è conv
          dim:     se type è conv, indica le dimensioni dei kernel
          stride: indica lo stride per i kernel
          padding: indica il padding da applicare all'input prima della convoluzione
        }
    %}
    
    SIGMA = 0.1;
    net.weights = {};
    net.biases = {};
    net.activations = {};
    net.deriv_func = {};
    net.layers = layers;
    net.n_layers = n_layers;
    
    for i=2 : n_layers
       if layers{i}.type == 2   %Livello convoluzionale
           net.weights{i-1} = flatten_kernel(randn(layers{i}.dim(1), layers{i}.dim(2), layers{i}.n_neurons));
           net.layers{i}.output_shape = size(flatten_input(randn(net.layers{i-1}.dim), net.layers{i}.dim(1), net.layers{i}.dim(2), net.layers{i}.stride, net.layers{i}.padding)*net.weights{i-1});
       elseif layers{i}.type == 1   %Livello full connected
           if layers{i-1}.type == 2         %se il layer precedente è convoluzionale allora devo cambiare i pesi
                net.weights{i-1} = SIGMA*randn(layers{i}.n_neurons, net.layers{i-1}.output_shape(1));
           else
                net.weights{i-1} = SIGMA*randn(layers{i}.n_neurons, layers{i-1}.n_neurons); 
           end
           
           net.layers{i}.output_shape = size(net.weights{i-1} * randn(net.layers{i-1}.output_shape));    
       end
       
       if net.layers{i}.type == 2
            net.biases{i-1} = SIGMA*randn(layers{i}.n_neurons, 1)';%gpuArray(SIGMA*randn(n_neurons(i), 1));
       else
            net.biases{i-1} = SIGMA*randn(layers{i}.n_neurons, 1);
       end
       net.activations{i-1} = actv_functions{i-1};
       net.deriv_func{i-1} = deriv_func{i-1};
    end
    
    %net.activations{n_layers} = actv_functions{n_layers};
    %net.deriv_func{n_layers} = deriv_func{n_layers};
end

