function new_momentum = updateMomentum(net, currentMomentum,old_W_deriv, current_W_deriv)
etaPlus = 1.2;
etaMinus = 0.5;
etaMax = 50;
etaMin = 0;
new_momentum = zeros(size(currentMomentum,1), size(currentMomentum,2));

res = current_W_deriv .* old_W_deriv;
%{
disp('current mom');
disp(currentMomentum);
disp((res>0));
disp((res>0) .* currentMomentum);
%}
new_momentum = new_momentum + (etaPlus*((res>0) .* currentMomentum));
new_momentum = new_momentum + (-etaMinus*((res<0) .* currentMomentum));
new_momentum = new_momentum + ((res==0) .* currentMomentum);
%{
disp('new mom');
disp(new_momentum);

disp('current deriv');
disp(current_W_deriv);

disp( 'old deriv');
disp( old_W_deriv);

disp('res');
disp( res );
%}
end

