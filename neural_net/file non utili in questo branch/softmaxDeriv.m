function e = softmaxDeriv(x)
    s = softmax_function(x);
    e = diag(s) - s*s';
end

