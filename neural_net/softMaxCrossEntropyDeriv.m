function [e] =softMaxCrossEntropyDeriv(Y,T)
   e=(exp(Y')./sum(exp(Y'),2)) - T;
   e=e';
end

