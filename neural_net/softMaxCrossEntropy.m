function [e] = softMaxCrossEntropy(Y,T)
    e = mean(-sum(T .* (log(exp(Y') ./ sum(exp(Y'), 2))), 2));
    %e = -sum(sum(T.*Y',2) + log(exp(Y') ./ sum(exp(Y'), 2)), 1);
end

