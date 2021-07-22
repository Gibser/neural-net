%% Load Data
clear;
load('X.mat');
load('Y.mat');

XT = X(:,:,1:128);
YT = Y(1:128);

XV = X(:,:,129:256);
YV = Y(129:256);

build_Y(YT);
build_Y(YV);
%% Load layers

load('layers2.mat');
layers2{1}.dim=[28 28 128];
layers2{3}.n_neurons=10;

%% load net
net = net_conv_FC(layers2, {@sigmoid, @sigmoid}, {@sigmoidDeriv, @sigmoidDeriv}, 3);
[err, final_net, err_val] = learningPhase_convFC(net, 100, XT, YT, XV, YV, @crossEntropyMC, @crossEntropyMCDeriv, 2, 0.01, 0.8, 128);


%TEST
% [a,z] = forward_step_convFC(final_net, X(:,:,1));
% z{2}
% Y(1);
