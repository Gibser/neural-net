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
    %net.biases = {};
    net.activations = {};
    net.deriv_func = {};
    net.layers = layers;
    net.n_layers = n_layers;
    
    for i=2 : n_layers
       if layers{i}.type == 2   %Livello convoluzionale     
           if layers{i-1}.type ~= 0
                H_out = ((net.layers{i-1}.output_shape(1) + 2*net.layers{i}.padding - net.layers{i}.dim(1)) / net.layers{i}.stride) + 1;
                W_out = ((net.layers{i-1}.output_shape(2) + 2*net.layers{i}.padding - net.layers{i}.dim(2)) / net.layers{i}.stride) + 1;
                dimension = net.layers{i-1}.output_shape;
           else
                H_out = ((layers{i-1}.dim(1) + 2*net.layers{i}.padding - net.layers{i}.dim(1)) / net.layers{i}.stride) + 1;
                W_out = ((layers{i-1}.dim(2) + 2*net.layers{i}.padding - net.layers{i}.dim(2)) / net.layers{i}.stride) + 1;
                dimension = layers{i-1}.dim;
           end
           
           flat_input = flatten_input(randn(dimension), net.layers{i}.dim(1), net.layers{i}.dim(2), net.layers{i}.stride, net.layers{i}.padding);
           %he_uniform initialization for weights
           limit = sqrt(6 / size(flat_input, 1));       %limit = sqrt(6 / fan_in) (fan_in is the number of input units in the weight tensor)
           net.weights{i-1} = flatten_kernel(-limit + (limit+limit).*rand(layers{i}.dim(1), layers{i}.dim(2), layers{i}.n_neurons));
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           forward = flat_input*net.weights{i-1};
           net.layers{i}.output_shape = size(reshape(forward, H_out*W_out, net.layers{i}.n_neurons, []));
           net.layers{i}.feature_map_dim = [H_out W_out net.layers{i}.n_neurons];    %Dimensione feature map (vedi paper)
           net.layers{i}.H_out = H_out;
           net.layers{i}.W_out = W_out;
       elseif layers{i}.type == 1           %Livello full connected
           if layers{i-1}.type == 2         %se il layer precedente è convoluzionale allora devo cambiare i pesi
                %cioé: ogni kernel genera una feature map di
                %H_out*W_out*n_neurons in uscita, di conseguenza ogni
                %kernel ha un numero di neuroni pari alla stessa quantità,
                %cioé H_out*W_out*n_neurons. Quindi il livello denso
                %successivo dovrà ricevere connessioni da questo numero di
                %neuroni
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %he_uniform%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %limit = sqrt(6 / (net.layers{i-1}.feature_map_dim(1) * net.layers{i-1}.feature_map_dim(2) * net.layers{i-1}.n_neurons));
                %net.weights{i-1} = -limit + (limit+limit).*rand(layers{i}.n_neurons, net.layers{i-1}.feature_map_dim(1) * net.layers{i-1}.feature_map_dim(2) * net.layers{i-1}.n_neurons);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
                net.weights{i-1} = SIGMA*randn(layers{i}.n_neurons, net.layers{i-1}.feature_map_dim(1) * net.layers{i-1}.feature_map_dim(2) * net.layers{i-1}.n_neurons);
           else
                %he_uniform%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %limit = sqrt(6 / layers{i-1}.n_neurons);
                %net.weights{i-1} = -limit + (limit+limit).*rand(layers{i}.n_neurons, layers{i-1}.n_neurons); 
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                net.weights{i-1} = SIGMA*randn(layers{i}.n_neurons, layers{i-1}.n_neurons); 
           end
           %disp(size(net.weights{i-1}));
           %disp(net.layers{i-1}.output_shape);
           flatten = reshape(randn(net.layers{i-1}.output_shape), net.layers{i-1}.output_shape(1)*net.layers{i-1}.output_shape(2), []);
           net.layers{i}.output_shape = size(net.weights{i-1} * flatten);
       end
       %{
       if net.layers{i}.type == 2
            net.biases{i-1} = 0;%SIGMA*randn(net.layers{i}.output_shape(1), 1);%gpuArray(SIGMA*randn(n_neurons(i), 1));
       else
            net.biases{i-1} = SIGMA*randn(net.layers{i}.n_neurons, 1);
       end
       %}
       net.activations{i-1} = actv_functions{i-1};
       net.deriv_func{i-1} = deriv_func{i-1};
    end
    
    %net.activations{n_layers} = actv_functions{n_layers};
    %net.deriv_func{n_layers} = deriv_func{n_layers};
end
