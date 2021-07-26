function [a_, z_] = forward_step(net, x)
   z = x;
   for i=1 : net.n_layers-1
       a = net.weights{i} * z + net.biases{i};
       %disp(a);
       a_{i} = a;
       z = net.activations{i}(a);
       z_{i} = z;
   end
end

