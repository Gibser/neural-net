function net = RPROP(net, W_deriv, old_W_deriv, bias_deriv, epoch)
    %Aggiorna pesi e bias in base alla RPROP
    for i=1 : net.n_layers-1
        %disp(size(sign(W_deriv{i})));
        %disp(size(net.weightMomentums{i}));
        net.weights{i} = net.weights{i} - (sign(W_deriv{i}).*net.weightMomentums{i});
       % net.biases{i} = net.biases{i} - (sign(bias_deriv{i}).*net.biasMomentums{i});
        if( epoch==0 ) 
            net = gradientDescent(net, 0.0005, W_deriv,bias_deriv);
        else 
            net.weightMomentums{i} = updateMomentum( net, net.weightMomentums{i}, old_W_deriv{i}, W_deriv{i});
            %net.biasMomentums{i} = updateMomentum(net,net.biasMomentums{i}, old_W_deriv, W_deriv);
        end
   end
end

