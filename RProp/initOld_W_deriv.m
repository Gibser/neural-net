function old_W_deriv = initOld_W_deriv(net)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    for i=1 : net.n_layers-1
        old_W_deriv{i} = 0;
    end
end

