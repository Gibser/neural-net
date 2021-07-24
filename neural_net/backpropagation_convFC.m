function [W_deriv, bias_deriv] = backpropagation_convFC(net, x, t, derivFunErr)
    
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
    %decommentare per ricostruzione
    %delta_out = delta_out .* derivFunErr(reshape(z_{end}, 28, 28, []), t);
    %commentare per ric
    delta_out = delta_out .* derivFunErr(z_{end}, t);
    %delta_out = reshape(delta_out, 28*28, []);
    deltas{net.n_layers} = delta_out;
    %disp('delta out');
    %disp(size(delta_out));
    %disp(size(z_{end-1}'));
    W_deriv{net.n_layers-1} = delta_out * z_{end-1};
    %disp(W_deriv);
    w = 0; %Questo indice serve per iterare sulle matrici W della rete
    a = 1; %Questo indice serve per gli input a dei neuroni nei livelli
    z = 2;
    %bias_deriv{net.n_layers-1} = sum(delta_out, 2);
    bias_deriv{net.n_layers-1} = delta_out;
    %disp(size(delta_out));
    for i=net.n_layers-1 : -1: 2
        %disp(size(net.weights{end-w}));
        %disp(size(deltas{i+1}));
        if net.layers{i}.type == 2
            w_d = size(net.weights{end-w});
            dim_delta = size(deltas{i+1});
            dim_flatten = size(z_{end-z});
            deltas{i} = zeros(net.layers{i}.n_neurons, dim_flatten(1));
            c = 1;
            batch_size = dim_flatten(1) / (net.layers{i}.H_out*net.layers{i}.W_out);
            disp(batch_size);
            delta = zeros(net.layers{i}.n_neurons);
            for k=1 : net.layers{i}.n_neurons : w_d(2)
                for img=1 : batch_size
                    delta = net.weights{end-w}(:, k:k+net.layers{i}.n_neurons-1)' * deltas{i+1}(:, img);
                    delta = delta .* net.deriv_func{end-a}(a_{end-a}(img, k:k+net.layers{i}.n_neurons-1))';
                    %disp(size(delta));
                    deltas{i}(:, c) = deltas{i}(:, c) + delta;
                end
                    c = c + 1;
            end
  
            %disp(size(deltas{i}));
            %W_deriv{i-1} = zeros(size(deltas{i}(1, :)' * z_{end-z}(1, :)));
            W_deriv{i-1} = zeros(size(net.weights{i-1}));
            %disp(size(W_deriv{i-1}));
            %disp(size(z_{end-z}(1, :)));
            dim_flatten = size(z_{end-z});
            
            for k = 1 : dim_flatten(1)
                W_deriv{i-1} = W_deriv{i-1} + (deltas{i}(:, k) * z_{end-z}(k, :))';
            end
            
        else
            deltas{i} = net.weights{end-w}' * deltas{i+1};
            deltas{i} = deltas{i} .* net.deriv_func{end-a}(a_{end-a});
            W_deriv{i-1} = deltas{i} * z_{end-z}';
            if net.layers{i}.use_bias == 1
                bias_deriv{i-1} = deltas{i};
            end
        end
        %disp(size(deltas{i}));
        %disp(size(net.deriv_func{end-a}(a_{end-a})));
        %bias_deriv{i-1} = deltas{i};
        
        w = w + 1;
        a = a + 1;
        z = z + 1;
        %disp(W_deriv{1});
    end
    

end
