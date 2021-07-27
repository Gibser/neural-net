function [net, deltaW, deltaB] = GDMomentum(net, W_deriv, bias_deriv, Delta_prec, Delta_bias_prec, eta, momentum)
%% Gradient Descend con momento
% net - rete
% W_deriv - derivata della funzione di errore rispetto ai pesi
% Delta_prec - variazione dei pesi nell'iterazione precedente
% Delta_bias_prec - variazione dei bias nell'iterazione precedente
% eta - learning rate
% momentum - momento
% Aggiorna i pesi della rete utilizzando al tecnica della discesa del
% gradiente con momento    
for i=1 : net.n_layers-1
        deltaW{i} = eta*W_deriv{i} + momentum .* Delta_prec{i};
        net.weights{i} = net.weights{i} - deltaW{i};
        if net.layers{i+1}.use_bias == 1
           deltaB{i} = eta*bias_deriv{i} + momentum .* Delta_bias_prec{i};
           net.layers{i+1}.bias =  net.layers{i+1}.bias - deltaB{i};
        end
    end
end

