function e=crossEntropyMC(Y,T)
    e = -sum(sum(T' .* log(Y),1),2);
end