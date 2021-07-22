XTrain = X(:, :, 1:1024);
%XTrain = X(:, :, 1);
%YTrain = Y(1:128);
%XTrain_01 = (XTrain - min(XTrain(:))) / (max(XTrain(:)) - min(XTrain(:)));

XVal = X(:, :, 1025:2048);
%XVal = X(:, :, 2);
%XVal_01 = (XVal - min(XVal)) / (max(XVal) - min(XVal));
%YVal = Y(129:256);

%load('layers2.mat');

net = net_conv_FC(layers2, {@sigmoid, @sigmoid}, {@sigmoidDeriv, @sigmoidDeriv}, 3);
[err, final_net, err_val] = learningPhase_convFC(net, 1000, XTrain, XTrain, XVal, XVal, @MSE, @MSEDeriv, 2, 0.01, 0.0, 128);
