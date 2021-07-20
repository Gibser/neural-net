XTrain = X(:, :, 1:500);
YTrain = Y(1:128);

XVal = X(:, :, 501:1000);
YVal = Y(129:256);

%load('layers2.mat');

net = net_conv_FC(layers2, {@relu, @sigmoid}, {@reluDeriv, @sigmoidDeriv}, 3);
[err, final_net, err_val] = learningPhase_convFC(net, 15, XTrain, XTrain, XVal, XVal, @MSE, @MSEDeriv, 0.0001, 1);
