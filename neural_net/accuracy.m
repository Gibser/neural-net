function [acc] = accuracy(net, imgs, trueValues)
% Restituisce l'accuracy di n predizioni in input
% INPUT:
% net - rete
% imgs - dati da predirre
% trueValue - etichette vere dei dati da predirre
    [~, z] = forward_step_convFC(net, imgs);
    arr = softmax_function(z{end});
    [~, ind] = max(arr, [], 1);
    n_pred = sum((ind-1)==trueValues', 2);
    acc = (n_pred / length(trueValues))*100;
end

