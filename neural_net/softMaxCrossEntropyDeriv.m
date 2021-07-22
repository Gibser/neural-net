function [e] =softMaxCrossEntropyDeriv(Y,T)
   e=Y'-T;
   e=e';
end

