function y = sigmoidDeriv(x)
    z = sigmoid(x);
    y = z .* (1-z);
end