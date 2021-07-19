function [a_, z_] = forward_step_convFC(net, x)
   z = x;
   for i=1 : net.n_layers-1
       if net.layers{i}.type == 1
            a = net.weights{i} * flatten_input(z, net.layers{i}.dim(1), net.layers{i}.dim(2), net.layers{i}.stride, net.layers{i}.padding) + net.biases{i};
       else
            a = net.weights{i} * z + net.biases{i};
       end
       a_{i} = a;
       z = net.activations{i}(a);
       z_{i} = z;
   end
end

