function [e] = softMaxCrossEntropy(Y,T)
    e = mean((-sum(T .* (log(exp(Y') ./ sum(exp(Y'), 2))), 2)));
end

