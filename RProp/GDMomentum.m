function [net, deltaW] = GDMomentum(net, W_deriv, Delta_prec, eta, momentum)
    for i=1 : net.n_layers-1
        deltaW{i} = eta*W_deriv{i} + momentum .* Delta_prec{i};
        net.weights{i} = net.weights{i} - deltaW{i};
    end
end

