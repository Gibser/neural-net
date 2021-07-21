function e=crossEntropyMCDeriv(Y, T)
    e = -sum( T ./ Y , 2);
end