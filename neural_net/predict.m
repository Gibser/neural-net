function [] = predict(net,x)
[~,z]=forward_step_convFC(net,x);
getOutput(z{2});
end

