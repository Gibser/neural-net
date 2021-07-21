function [net, Delta, W_deriv] = RPROP(net, W_deriv, old_W_deriv, bias_deriv, epoch, Delta_prec)
    %Aggiorna pesi e bias in base alla RPROP
    %%%%
    %%
    etaPlus = 1.2;
    etaMinus = 0.5;
    deltaMax = 50;
	deltaMin = 1e-6; 
    for i=1 : net.n_layers-1
   % disp(['strato', num2str(i)]);
    %disp('Old Deriv');
    %disp(old_W_deriv{i});
    %disp('New Deriv');
    %disp(W_deriv{i});
        if( epoch==1 ) 
            disp('Prima iterazione');
            [net, Delta, ~]  = gradientDescent(net, 0.0001, W_deriv, bias_deriv);
        else 
            new_deltas = zeros(size(net.weightMomentums{i}));
            disp('W_deriv');
            disp(W_deriv);
            disp('old_W_deriv');
            disp(old_W_deriv);
            res = W_deriv{i} .* old_W_deriv{i};
            Delta{i} = zeros(size(net.weights{i}));
            newWeights{i} = zeros(size(net.weights{i}));
            disp('res');
            disp(res);
            new_deltas = new_deltas + ((res > 0) .* min(deltaMax, etaPlus .* net.weightMomentums{i}));
            new_deltas = new_deltas + ((res < 0) .* max(deltaMin, etaMinus .* net.weightMomentums{i}));
            new_deltas = new_deltas + ((res == 0) .* net.weightMomentums{i});
            disp('new_deltas');
            disp(new_deltas)
            Delta{i} = Delta{i} + (res > 0) .* (-sign(W_deriv{i}) .* new_deltas);
            Delta{i} = Delta{i} + (res < 0) .* Delta_prec{i};
            Delta{i} = Delta{i} + (res == 0) .* (-sign(W_deriv{i}) .* new_deltas);
            disp('Delta');
            disp(Delta);
            newWeights{i} = newWeights{i} + (res > 0) .* (net.weights{i} + Delta{i});
            newWeights{i} = newWeights{i} + (res == 0) .* (net.weights{i} + Delta{i});
            newWeights{i} = newWeights{i} + (res < 0) .* (net.weights{i} - Delta{i});
            
            W_deriv{i} = (res >= 0) .* W_deriv{i};
            
            net.weightMomentums{i} = new_deltas;
            net.weights{i} = newWeights{i};
            %net.weightMomentums{i} = updateMomentum(net, net.weightMomentums{i}, old_W_deriv{i}, W_deriv{i});
            %net.biasMomentums{i} = updateMomentum(net,net.biasMomentums{i}, old_W_deriv, W_deriv);
        end
        %net.weights{i} = net.weights{i} - (sign(W_deriv{i}).*net.weightMomentums{i});
        %net.biases{i} = net.biases{i} - (sign(bias_deriv{i}).*net.biasMomentums{i});
        
    end
   %%%%
end

