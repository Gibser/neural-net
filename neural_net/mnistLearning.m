XTrain = X(:, :, 1:100);
YTrain = Y(1:100);

XVal = X(:, :, 100:200);
YVal = Y(100:200);

net = net_conv_FC(layers, {@sigmoid, @sigmoid, @identity}, {@sigmoidDeriv, @sigmoidDeriv, @identityDeriv}, 3);
[err, final_net, err_val] = learningPhase_convFC(net, 100, XTrain, YTrain, XVal, YVal, @sumOfSquares, @sumOfSquaresDeriv, 0.001, 1);
