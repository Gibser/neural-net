function e=crossEntropyMC(Y,T)
    e = -sum(sum(T' .* log(Y),2));
end