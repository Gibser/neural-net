function new_momentum = updateMomentum(net,currentMomentum,old_W_deriv, current_W_deriv)
etaPlus = 1.2;
etaMinus = -0.9;
new_momentum = currentMomentum;
%disp(['W_deriv', current_W_deriv]);
%disp(size(old_W_deriv));
    for i=1 : net.n_layers-1
        res = current_W_deriv .* old_W_deriv;
        %disp(size(res{i}));
        [rows, cols] = size(res);
        for r=1 : rows
            for c=1: cols
                if(res(r,c)>0)
                    new_momentum(r,c) = etaMinus * currentMomentum(r,c);
                else
                    new_momentum(r,c) = etaPlus * currentMomentum(r,c);
                end
            end
        end

    end


end

