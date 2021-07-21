XTrain = X(:, :, 1:128);
YTrain = Y(1:128);

XVal = X(:, :, 129:256);
YVal = Y(129:256);

%load('layers2.mat');

net = net_conv_FC(layers2, {@relu, @sigmoid}, {@reluDeriv, @sigmoidDeriv}, 3);
[err, final_net, err_val] = learningPhase_convFC(net, 1000, XTrain, XTrain, XVal, XVal, @MSE, @MSEDeriv, 0.001, 1);
