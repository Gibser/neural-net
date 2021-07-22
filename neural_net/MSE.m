function E=MSE(y,t)
    D = abs(t-y').^2;
    E = sum(D(:))/numel(t);
end

