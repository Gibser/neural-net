function [net, sum_of_weights_gradients, sum_of_bias_gradients] = adagrad(net, W_deriv, bias_deriv, eta, epsilon, sum_of_weights_gradients, sum_of_bias_gradients)
    for i=1 : net.n_layers-1
        sum_of_weights_gradients{i} = W_deriv{i} .^ 2;
        deltaW = W_deriv{i} ./ (epsilon + sqrt(sum_of_weights_gradients{i}));
        net.weitghts{i} = net.weights{i} - eta*deltaW;
        
        if net.layers{i+1}.use_bias == 1
           sum_of_bias_gradients{i} = bias_deriv{i} .^ 2;
            deltaB = bias_deriv{i} ./ (epsilon + sqrt(sum_of_bias_gradients{i}));
            net.layers{i+1}.bias = net.layers{i+1}.bias - eta*deltaB;
        end
    end
end

