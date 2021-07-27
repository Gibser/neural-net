function predict(net,x)
% Restituisce la predizione di una rete
% net - rete
% x - immagine da classificare
    [~,z]=forward_step_convFC(net,x);
    getOutput(z{end});
end

