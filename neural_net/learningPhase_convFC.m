function [err, final_net, err_val, acc_tr, acc_val] = learningPhase_convFC(net, N, x, t, x_val, t_val, errFunc, errFuncDeriv, BATCH, eta, momentum, batch_size)
%Funzione che eseguie la fase di learning per la rete
% net - rete
% N - numero epoche
% x - training set
% t - target del training set
% x_val - validation set
% t_val - target validation set
% errFunc - funzione di errore
% errFunDeri - derivata della funzione di errore
% BATCH - learning di tipo (1) ONLINE, (2) BATCH, (3) MINI-BATCH
% eta - learning rate
% momentum - momento
% batch_size - grandezza del batch

err = zeros(1,N);
err_val = zeros(1,N);
[~, z_] = forward_step_convFC(net, x_val);
%y_val = reshape(z_{end}, 28, 28, []); decommentare per ricostruzione immagini
y_val=z_{end}; %commentare per ricostruzione immagini
%disp(size(y_val));
%disp(size(t_val));
min_err = errFunc(y_val, t_val);
final_net = net;
old_Deltas = {};
old_Deltas_bias = {};
sum_of_weights_gradients = {};
sum_of_bias_gradients = {};

%%Imposto tutti i delta precedenti a zero (precedentemente alla prima
%%epoca, non ci sono dei delta quindi li impostiamo a zero)
for i=1 : net.n_layers-1
    old_Deltas{i} = zeros(size(net.weights{i}));
    sum_of_weights_gradients{i} = zeros(size(net.weights{i}));
    if net.layers{i+1}.use_bias == 1
        old_Deltas_bias{i} = zeros(size(net.layers{i+1}.bias));
        sum_of_bias_gradients{i} = zeros(size(net.layers{i+1}.bias));
    end
end


for epoch=1:N 
    ind=randperm(size(x,3));
    x = x(:, :, ind);
    t=t(ind,:);
    disp('==================================');
    disp(['Epoca corrente:' 9, 9, 9, num2str(epoch)]);
    if BATCH==0
    %% ONLINE LEARNING
        for n=1:size(x,3)
            [w, b] = backpropagation_convFC(net, x(:,:,n), t(n, :), errFuncDeriv);
            [net, oldDeltas, old_Deltas_bias] = GDMomentum(net, w, b, old_Deltas, old_Deltas_bias, eta, momentum);
        end
    elseif BATCH==1
    %% BATCH LEARNING
     [w, b] = backpropagation_convFC(net, x, t, errFuncDeriv);
     [net, oldDeltas, old_Deltas_bias] = GDMomentum(net, w, b, old_Deltas, old_Deltas_bias, eta, momentum); 
    elseif BATCH==2
    %% MINI-BATCH
        c = 1;
        % Definisco gli intervalli dei mini-batch
        for j = 1 : batch_size : length(x)
           intervals(:, c) = [j j+batch_size-1];
           c = c + 1;
        end
        permutations = randperm(size(intervals, 2), size(intervals, 2));
        int_perm = intervals(:, permutations);
        if batch_size ~= length(x)
            for k=1 : length(int_perm)
                indexes = int_perm(:, k);  %Permutazione casuale dei batch
                x_batch = x(:, :, indexes(1):indexes(2));
                t_batch = t(indexes(1):indexes(2),:);

                indexes_perm = randperm(batch_size, batch_size);%Permutazione casuale delle immagini NEL batch
                %Solo permutazione batch
                [w, b] = backpropagation_convFC(net, x(:, :, indexes(1):indexes(2)), t(indexes(1):indexes(2),:), errFuncDeriv);
                %Permutazione batch + permutazione immagini nel batch
                [net, old_Deltas, old_Deltas_bias] = GDMomentum(net, w, b, old_Deltas, old_Deltas_bias, eta, momentum);       
            end  
        else
            [w, b] = backpropagation_convFC(net, x, t, errFuncDeriv);

        end
        [net, old_Deltas, old_Deltas_bias] = GDMomentum(net, w, b, old_Deltas, old_Deltas_bias, eta, momentum);
    end
    [~, z2_] = forward_step_convFC(net, x);
    [~, z_] = forward_step_convFC(net, x_val);

    y=z2_{end}; 
    y_val= z_{end}; 
    err(epoch) = errFunc(y, t); 
    err_val(epoch) = errFunc(y_val, t_val);
    acc_tr(epoch)=accuracy(final_net, x, vector_class_to_int_class(t));
    acc_val(epoch)=accuracy(final_net,x_val, vector_class_to_int_class(t_val));
    disp(['Errore di training:', 9 9 num2str(err(epoch))]);
    disp(['Errore di validazione:',9 num2str(err_val(epoch))]);
    disp(['Accuracy di training: ',9 num2str(accuracy(final_net, x, vector_class_to_int_class(t))),'%']);
    disp(['Accuracy di validation: ', num2str(accuracy(final_net,x_val, vector_class_to_int_class(t_val))), '%']);
    
    %% Aggiornamento della rete migliore
    if err_val(epoch)< min_err
        min_err=err_val(epoch);
        final_net = net;
    end
end
end