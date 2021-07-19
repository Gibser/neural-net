XTrain = X(:, :, 1:100);
YTrain = Y(1:100);

XVal = X(:, :, 100:200);
YVal = Y(100:200);

load('layers.mat');

net = net_conv_FC(l, {@sigmoid, @sigmoid, @identity}, {@sigmoidDeriv, @sigmoidDeriv, @identityDeriv}, 4);
[err, final_net, err_val] = learningPhase_convFC(net, 100, XTrain, YTrain, XVal, YVal, @sumOfSquares, @sumOfSquaresDeriv, 0.001, 1);
