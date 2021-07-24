function [acc] = accuracy(net, imgs, trueValues)
% Restituisce l'accuracy di n predizioni in input
% INPUT:
% net - rete
% imgs - dati da predirre
% trueValue - etichette vere dei dati da predirre
    %arr = softmax(Y);
    n_pred = 0;
    for i=1 : length(trueValues)
        arr = forward_step_convFC(net, imgs(:, :, i));
        %disp(softmax(arr{end}));
        %disp(T(i));
        %[m, ind] = max(arr(:, i));
        [m, ind] = max(softmax(arr{end}));
        n_pred = n_pred + (ind-1==trueValues(i));
    end
    acc = (n_pred / length(trueValues))*100;
    %disp([num2str(acc), '%']);
end

