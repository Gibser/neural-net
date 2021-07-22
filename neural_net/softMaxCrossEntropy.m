function [e] = softMaxCrossEntropy(Y,T)
    e = -sum(sum(T .* log(exp(Y')/sum(exp(Y'),1)),1),2);
end

