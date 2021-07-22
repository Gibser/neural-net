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
final_net = net;
old_Deltas = {};

for i=1 : net.n_layers-1
    old_Deltas{i} = zeros(size(net.weights{i}));
end

for epoch=1:N %In QUESTO CASO sto supponendo di fare sempre tutte le iterazioni
    %LEARNING ON-LINE
    ind=randperm(size(x,3));
    x = x(:, :, ind);
    %t = t(:, :, ind); decommentare per ricostruzione
    %disp(size(t));
    t=t(ind,:); %decommentare per ricostruzione
    disp(['epoch:' num2str(epoch)]);
    if BATCH==0
        for n=1:size(x,2)
            [w] = backpropagation_convFC(net, x(:,n), t(:,n), errFuncDeriv);
            %QUESTA REGOLA DI AGGIORNAMENTO SI PUO' SCEGLIERE
            net = gradientDescent_convFC(net, eta, w);
        end
    elseif BATCH==1
     %BATCH LEARNIG
     [w] = backpropagation_convFC(net, x, t, errFuncDeriv);
     %QUESTA REGOLA DI AGGIORNAMENTO SI PUO' SCEGLIERE
     net = GDMomentum(net, w, old_Deltas, eta, momentum);
    elseif BATCH==2
        for k=1 : batch_size : size(x, 2)
            rand_index = int32((batch_size - 1) .* rand(1) + (1));
            %disp(rand_index);
            %[w] = backpropagation_convFC(net, x(:, :, k:k+batch_size-1), t(:, :, k:k+batch_size-1), errFuncDeriv);
            %[w] = backpropagation_convFC(net, x(:, :, k:k+batch_size-1), t(k:k+batch_size-1,:), errFuncDeriv);
            %[w] = backpropagation_convFC(net, x(:, :, k + rand_index), t(:, :, k + rand_index), errFuncDeriv); %SGD
            [w] = backpropagation_convFC(net, x(:, :, k + rand_index), t(k + rand_index,:), errFuncDeriv); %SGD
            [net, old_Deltas] = GDMomentum(net, w, old_Deltas, eta, momentum); 
        end       
    end
    
    [~, z2_] = forward_step_convFC(net, x);
    [~, z_] = forward_step_convFC(net, x_val);
    
    %y = reshape(z2_{end}, 28, 28, []); decommentare per ric
    y=z2_{end}; %commentare per ric
    %y_val = reshape(z_{end}, 28, 28, []); decommentare per ric
    y_val= z_{end}; %commentare per ric
    err(epoch) = sum(errFunc(y,t)); 
    disp(size(y));
    disp(size(t));
    err_val(epoch) = sum(errFunc(y_val,t_val));
    disp(['err train:' num2str(err(epoch)) ' err val:' num2str(err_val(epoch))]);
    if err_val(epoch)< min_err
        min_err=err_val(epoch);
        final_net = net;
    end
end
end