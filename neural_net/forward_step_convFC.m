function [a_, z_] = forward_step_convFC(net, x)
   z = x;
   for i=2 : net.n_layers
       if net.layers{i}.type == 2
            a = flatten_input(z, net.layers{i}.dim(1), net.layers{i}.dim(2), net.layers{i}.stride, net.layers{i}.padding)*net.weights{i-1};% + net.biases{i-1};
            a_dim = size(a);
            a = reshape_flatten(a, net.layers{i}.n_neurons, net.layers{i}.H_out*net.layers{i}.W_out);
       else
            if net.layers{i-1}.type == 2
                forward = z;
                a = net.weights{i-1} * forward;
            else
                a = net.weights{i-1} * z;
            end
            
            if net.layers{i}.use_bias == 1
               a = a + net.layers{i}.bias; 
            end
       end
       a_{i-1} = a;
       z = net.activations{i-1}(a);
       z_{i-1} = z;
   end
end

