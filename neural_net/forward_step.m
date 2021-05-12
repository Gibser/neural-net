function out = forward_step(net, x)
   z = x;
   for i=1 : net.n_layers-1
       a = net.weights{i} * z + net.biases{i};
       disp(a);
       z = net.activations{i}(a);
   end
end

