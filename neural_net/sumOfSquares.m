function E=sumOfSquares(y,t)
%function e=sumOfSquares(y,t)
    E = (1/2) * sum(sum((y-t) .^ 2));
end