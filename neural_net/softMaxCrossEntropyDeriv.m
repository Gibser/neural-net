function [e] =softMaxCrossEntropyDeriv(Y,T)
   e=sum(Y'-T, 2);
end

