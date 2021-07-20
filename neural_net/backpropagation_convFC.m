function [W_deriv] = backpropagation_convFC(net, x, t, derivFunErr)
    
    W_deriv = {};
    deltas = {};
    bias_deriv = {};
    %% FASE FORWARD-PROPAGATION
    [a_, z_] = forward_step_convFC(net, x);
    %z_ = [{reshape(x, 28*28, [])} z_(:)']; %Sostituire reshape(x, 28*28, []) con flatten_input(XTrain...)%
    z_ = [{flatten_input(x, net.layers{2}.dim(1), net.layers{2}.dim(2), net.layers{2}.stride, net.layers{2}.padding)} z_];
    
    %% FASE BACK-PROPAGATION (calcolo delta)
    %Calcolo dela nodi di uscita
    delta_out = net.deriv_func{end}(a_{end});
    delta_out = delta_out .* derivFunErr(reshape(z_{end}, 28, 28, []), t);
    %delta_out = reshape(delta_out, 28*28, []);
    deltas{net.n_layers} = delta_out;
    W_deriv{net.n_layers-1} = delta_out * z_{end-1}';
    %disp(W_deriv);
    w = 0; %Questo indice serve per iterare sulle matrici W della rete
    a = 1; %Questo indice serve per gli input a dei neuroni nei livelli
    z = 2;
    %bias_deriv{net.n_layers-1} = sum(delta_out, 2);
    %bias_deriv{net.n_layers-1} = delta_out;
    %disp(size(delta_out));
    for i=net.n_layers-1 : -1: 2
        %disp(size(net.weights{end-w}));
        %disp(size(deltas{i+1}));
        if net.layers{i}.type == 2
            w_d = size(net.weights{end-w});
            dim_delta = size(deltas{i+1});
            deltas{i} = zeros(w_d(2), dim_delta(2));
            for k=1 : w_d(2)
                deltas{i}(k, :) = net.weights{end-w}(:, k)' * deltas{i+1};
                deltas{i}(k, :) = deltas{i}(k, :) .* net.deriv_func{end-a}(a_{end-a}(k, :));
            end
            
            W_deriv{i-1} = zeros(size(deltas{i}(1, :)' * z_{end-z}(1, :)));
            for k=1 : w_d(2)
                W_deriv{i-1} = W_deriv{i-1} + deltas{i}(k, :)' * z_{end-z}(k, :);
            end
            
            W_deriv{i-1} = W_deriv{i-1}';
        else
            deltas{i} = net.weights{end-w}' * deltas{i+1};
            deltas{i} = deltas{i} .* net.deriv_func{end-a}(a_{end-a});
            W_deriv{i-1} = deltas{i} * z_{end-z}';
        end
        %disp(size(deltas{i}));
        %disp(size(net.deriv_func{end-a}(a_{end-a})));
        %bias_deriv{i-1} = deltas{i};
        
        w = w + 1;
        a = a + 1;
        z = z + 1;
        %disp(W_deriv);
    end
    

end
