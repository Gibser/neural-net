function net = RPROP(net, W_deriv, old_W_deriv, bias_deriv, epoch)
    %Aggiorna pesi e bias in base alla RPROP
    %%%%
    %%
    for i=1 : net.n_layers-1
   % disp(['strato', num2str(i)]);
    %disp('Old Deriv');
    %disp(old_W_deriv{i});
    %disp('New Deriv');
    %disp(W_deriv{i});
        if( epoch==1 ) 
            disp('Prima iterazione');
            net = gradientDescent(net, 0.0001, W_deriv,bias_deriv);
        else 
            net.weightMomentums{i} = updateMomentum(net, net.weightMomentums{i}, old_W_deriv{i}, W_deriv{i});
            %net.biasMomentums{i} = updateMomentum(net,net.biasMomentums{i}, old_W_deriv, W_deriv);
        end
        net.weights{i} = net.weights{i} - (sign(W_deriv{i}).*net.weightMomentums{i});
        %net.biases{i} = net.biases{i} - (sign(bias_deriv{i}).*net.biasMomentums{i});
        
    end
   %%%%
end

