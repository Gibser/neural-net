function e=crossEntropyMC(Y,T)
    disp(size(T));
    disp(size(Y));
    e = -sum(sum(T .* log(Y'),1),2);
   
end