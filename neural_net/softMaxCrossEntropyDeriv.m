function [e] =softMaxCrossEntropyDeriv(Y,T)
   e=sum(Y-T);
end

