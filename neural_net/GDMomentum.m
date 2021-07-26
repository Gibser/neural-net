function [net, deltaW] = GDMomentum(net, W_deriv, Delta_prec, eta, momentum)
%% Gradient Descend con momento
% net - rete
% W_deriv - derivata della funzione di errore rispetto ai pesi
% Delta_prec - variazione del peso nell'iterazione precedente
% eta - learning rate
% momentum - momento
% Aggiorna i pesi della rete utilizzando al tecnica della discesa del
% gradiente con momento
    for i=1 : net.n_layers-1
        deltaW{i} = eta*W_deriv{i} + momentum .* Delta_prec{i};
        net.weights{i} = net.weights{i} - deltaW{i};
    end
end

