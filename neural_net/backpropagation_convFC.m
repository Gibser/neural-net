function [W_deriv, bias_deriv] = backpropagation_convFC(net, x, t, derivFunErr)
    
    W_deriv = {};
    deltas = {};
    bias_deriv = {};
    %% FASE FORWARD-PROPAGATION
    [a_, z_] = forward_step_convFC(net, x);
    z_ = [{x} z_(:)'];
    
    %% FASE BACK-PROPAGATION (calcolo delta)
    %Calcolo dela nodi di uscita
    delta_out = net.deriv_func{end}(a_{end});
    delta_out = delta_out .* derivFunErr(z_{end}, t);
    deltas{net.n_layers-1} = delta_out;
    W_deriv{net.n_layers-1} = delta_out * z_{end-1}';
    %disp(W_deriv);
    w = 0; %Questo indice serve per iterare sulle matrici W della rete
    a = 1; %Questo indice serve per gli input a dei neuroni nei livelli
    z = 2;
    bias_deriv{net.n_layers-1} = sum(delta_out, 2);
    for i=net.n_layers-2 : -1: 1
        %disp(deltas{i+1});
        if net.layers{i}.type == 2
            disp(size(net.weights{end-w}'));
            disp(size(deltas{i+1}'))
            deltas{i} = net.weights{end-w}' * deltas{i+1}';
        else
            deltas{i} = net.weights{end-w}' * deltas{i+1};
        end
        deltas{i} = deltas{i} .* net.deriv_func{end-a}(a_{end-a});
        W_deriv{i} = deltas{i} * z_{end-z}';
        bias_deriv{i} = sum(deltas{i}, 2);
        w = w + 1;
        a = a + 1;
        z = z + 1;
        %disp(W_deriv);
    end
    

end
