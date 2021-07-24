function [acc] = accuracy(net, imgs, T)
    %arr = softmax(Y);
    n_pred = 0;
    for i=1 : length(T)
        arr = forward_step_convFC(net, imgs(:, :, i));
        disp(softmax(arr{end}));
        disp(T(i));
        %[m, ind] = max(arr(:, i));
        [m, ind] = max(softmax(arr{end}));
        n_pred = n_pred + (ind-1==T(i));
    end
    acc = n_pred / length(T);
end

