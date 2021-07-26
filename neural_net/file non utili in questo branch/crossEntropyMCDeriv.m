function e=crossEntropyMCDeriv(Y, T)
   % disp(size(T));
    %disp(size(Y));
    e = -sum( T' ./ Y , 2);
end