function [err, final_net, err_val] = learningPhase2(net, N, x, t, x_val, t_val, errFunc, errFuncDeriv, BATCH)
%Learning rate e' un parametro che posso scegliere, e relativo alla regola di aggiornamento
% scelta, ed è considerato un iper-parametro del processo di
% learning. In genere passato come parametro alla funzione.
%Apprendimento Batch (1) oppure on-line (0). In genere come parametro della funzione
err = zeros(1,N);
err_val = zeros(1,N);
[~, z_] = forward_step(net, x_val);
y_val = z_{end};    %Questo reshape va generalizzato nella struttura della rete
min_err = errFunc(y_val, t_val);%sumOfSquares(y_val, t_val)
final_net = net;
old_W_deriv=initOld_W_deriv(net);
for epoch=1:N %In QUESTO CASO sto suppenendo di fare sempre tutte le iterazioni
    %LEARNING ON-LINE
    if BATCH==0
        for n=1:size(x,2)
            [w, b] = backpropagation(net, x(:,n), t(:,n), errFuncDeriv);
            %QUESTA REGOLA DI AGGIORNAMENTO SI PUO' SCEGLIERE
            net = RPROP(net, w, old_W_deriv, b, epoch);
            old_W_deriv = w;
        end
    else
        %BATCH LEARNIG
        [w, b] = backpropagation(net, x, t, errFuncDeriv);
        %QUESTA REGOLA DI AGGIORNAMENTO SI PUO' SCEGLIERE
        
        net = RPROP(net, w, old_W_deriv, b, epoch);
        old_W_deriv = w;
    end
    
    [~, z2_] = forward_step(net, x);
    [~, z_] = forward_step(net, x_val);
    y = z2_{end};
    y_val = z_{end};
    err(epoch) = errFunc(y,t);
    err_val(epoch) = errFunc(y_val,t_val);
    disp(['err train:' num2str(err(epoch)) ' err val:' num2str(err_val(epoch))]);
    if err_val(epoch)< min_err
        min_err=err_val(epoch);
        final_net = net;
    end
end
end