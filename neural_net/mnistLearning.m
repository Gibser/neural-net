XTrain = X(:, :, 1:128);
YTrain = Y(1:128);

XVal = X(:, :, 129:256);
YVal = Y(129:256);

load('layers2.mat');

net = net_conv_FC(layers2, {@relu, @relu}, {@reluDeriv, @reluDeriv}, 3);
[err, final_net, err_val] = learningPhase_convFC(net, 10, XTrain, XTrain, XVal, XVal, @sumOfSquares, @sumOfSquaresDeriv, 0.001, 1);
