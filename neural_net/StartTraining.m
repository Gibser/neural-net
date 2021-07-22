%% Load Data
clear;
load('X.mat');
load('Y.mat');

XT = X(:,:,1:128);
YT = Y(1:128);

XV = X(:,:,129:256);
YV = Y(129:256);

YT=build_Y(YT);
YV=build_Y(YV);
%% Load layers

load('layers2.mat');
layers2{1}.dim=[28 28 1];
layers{2}.n_neurons=128;
layers2{3}.n_neurons=10;

%% load net
net = net_conv_FC(layers2, {@relu, @sigmoid}, {@reluDeriv, @sigmoidDeriv}, 3);
[err, final_net, err_val] = learningPhase_convFC(net, 1000, XT, YT, XV, YV, @softMaxCrossEntropy, @softMaxCrossEntropyDeriv, 2, 0.001, 0, 128);


%TEST
% [a,z] = forward_step_convFC(final_net, X(:,:,1));
% z{2}
% Y(1);
