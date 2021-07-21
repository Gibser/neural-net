function new_momentum = updateMomentum(net, currentMomentum,old_W_deriv, current_W_deriv)
etaPlus = 1.2;
etaMinus = -0.5;
etaMax = 50;
etaMin = 0;
new_momentum = zeros(size(currentMomentum,1), size(currentMomentum,2));

res = current_W_deriv .* old_W_deriv;

[rows, cols] = size(res);
%disp('current mom');
%disp(currentMomentum);
for r=1 : rows
    for c=1: cols
        if(res(r,c)>0)
            new_momentum(r,c) = etaPlus * currentMomentum(r,c);
            %new_momentum(r,c) = min(etaPlus * currentMomentum(r,c), etaMax);
        else
            new_momentum(r,c) = etaMinus * currentMomentum(r,c);
            %new_momentum(r,c) = max(etaMinus * currentMomentum(r,c), etaMin);
        end

    end
end
%disp('new mom');
%disp(new_momentum);

%disp('current deriv');
%disp(current_W_deriv);

%disp( 'old deriv');
%disp( old_W_deriv);

%disp('res');
%disp( res );
end

