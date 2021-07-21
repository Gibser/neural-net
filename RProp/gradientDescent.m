function [net, deltaW, deltaB] = gradientDescent(net, eta, W_deriv, bias_deriv)
    for i=1 : net.n_layers-1
        net.weights{i} = net.weights{i} - eta*W_deriv{i};
        net.biases{i} = net.biases{i} - eta*bias_deriv{i};
        deltaW{i} = eta*W_deriv{i};
        deltaB{i} = eta*bias_deriv{i};
    end
end