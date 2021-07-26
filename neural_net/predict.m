function predict(net,x)
[~,z]=forward_step_convFC(net,x);
getOutput(z{end});
end

