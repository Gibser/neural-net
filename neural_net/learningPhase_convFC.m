function [err, final_net, err_val] = learningPhase_convFC(net, N, x, t, x_val, t_val, errFunc, errFuncDeriv, BATCH, eta, momentum, batch_size)
         %Learning rate e' un parametro che posso scegliere, e relativo alla regola di aggiornamento
         % scelta, ed è considerato un iper-parametro del processo di
         % learning. In genere passato come parametro alla funzione.
         %Apprendimento Batch (1) oppure on-line (0). In genere come parametro della funzione
err = zeros(1,N);
err_val = zeros(1,N);
[~, z_] = forward_step_convFC(net, x_val);
%y_val = reshape(z_{end}, 28, 28, []); decommentare per ricostruzione immagini
y_val=z_{end}; %commentare per ricostruzione immagini
%disp(size(y_val));
%disp(size(t_val));
min_err = errFunc(y_val, t_val);
disp(min_err);
final_net = net;
old_Deltas = {};
old_Deltas_bias = {};

for i=1 : net.n_layers-1
    old_Deltas{i} = zeros(size(net.weights{i}));
    if net.layers{i+1}.use_bias == 1
        old_Deltas_bias{i} = zeros(size(net.layers{i+1}.bias));
    end
end

disp(old_Deltas_bias);
for epoch=1:N %In QUESTO CASO sto supponendo di fare sempre tutte le iterazioni
    %LEARNING ON-LINE
    ind=randperm(size(x,3));
    x = x(:, :, ind);
    %t = t(:, :, ind); decommentare per ricostruzione
    %disp(size(t));
    t=t(ind,:); %decommentare per ricostruzione
    disp(['epoch:' num2str(epoch)]);
    if BATCH==0
        for n=1:size(x,3)
            [w] = backpropagation_convFC(net, x(:,:,n), t(n, :), errFuncDeriv);
            %QUESTA REGOLA DI AGGIORNAMENTO SI PUO' SCEGLIERE
            net = GDMomentum(net, w, old_Deltas, oldDeltas_bias, eta, momentum);
        end
    elseif BATCH==1
     %BATCH LEARNIG
     [w, b] = backpropagation_convFC(net, x, t, errFuncDeriv);
     %QUESTA REGOLA DI AGGIORNAMENTO SI PUO' SCEGLIERE
     [net, oldDeltas, old_Deltas_bias] = GDMomentum(net, w, b, old_Deltas, old_Deltas_bias, eta, momentum);
    elseif BATCH==2
        for k=1 : batch_size : size(x, 3)
            %rand_index = floor((batch_size - 1) .* rand(1) + (1));
            %disp(rand_index);
            %[w] = backpropagation_convFC(net, x(:, :, k:k+batch_size-1), t(:, :, k:k+batch_size-1), errFuncDeriv);
            %[w] = backpropagation_convFC(net, x(:, :, k:k+batch_size-1), t(k:k+batch_size-1,:), errFuncDeriv);
            %[w] = backpropagation_convFC(net, x(:, :, k + rand_index), t(:, :, k + rand_index), errFuncDeriv); %SGD
            %[w, b] = backpropagation_convFC(net, x(:, :, k + rand_index), t(k + rand_index,:), errFuncDeriv); %SGD
            [w, b] = backpropagation_convFC(net, x(:, :, k:k+batch_size-1), t(k:k+batch_size-1,:), errFuncDeriv);
            [net, old_Deltas, old_Deltas_bias] = GDMomentum(net, w, b, old_Deltas, old_Deltas_bias, eta, momentum); 
        end       
    end
    
    %rand_index = floor((length(x) - 1) .* rand(1) + (1));
    %rand_index_val = floor((length(x_val) - 1) .* rand(1) + (1));
    [~, z2_] = forward_step_convFC(net, x);
    [~, z_] = forward_step_convFC(net, x_val);
    
    %y = reshape(z2_{end}, 28, 28, []); decommentare per ric
    y=z2_{end}; %commentare per ric
    %y_val = reshape(z_{end}, 28, 28, []); decommentare per ric
    y_val= z_{end}; %commentare per ric
    err(epoch) = errFunc(y, t); 
    err_val(epoch) = errFunc(y_val, t_val);
   
    disp(['err train:' num2str(err(epoch)) 9 ' err val:' num2str(err_val(epoch))]);
    disp(['accuracy on train: ',num2str(accuracy(final_net,x,t)),'%' , 9 'accuracy on val: ' num2str(accuracy(final_net,x_val, t_val )), '%']);
   disp('______________________________');
    if err_val(epoch)< min_err
        min_err=err_val(epoch);
        final_net = net;
    end
end
end