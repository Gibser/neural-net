function e = crossEntropy(y, t)
    e = -(t .* log(y) + (1+t) .* log(1-y));
end

