function E=MSEDeriv(y,t)
    D = -2 * (t-y);
    E = sum(D(:))/numel(t);
end

