function [net, deltaW, deltaB] = GDMomentum(net, W_deriv, bias_deriv, Delta_prec, Delta_bias_prec, eta, momentum)
    for i=1 : net.n_layers-1
        deltaW{i} = eta*W_deriv{i} + momentum .* Delta_prec{i};
        net.weights{i} = net.weights{i} - deltaW{i};
        if net.layers{i+1}.use_bias == 1
           deltaB{i} = eta*bias_deriv{i} + momentum .* Delta_bias_prec{i};
           net.layers{i+1}.bias =  net.layers{i+1}.bias - deltaB{i};
        end
    end
end

