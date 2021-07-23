function e = crossentropyDeriv(softmax_input, T)
    e = softmax_input - T;
end

