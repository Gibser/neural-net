clear;


TRAIN_SIZE = 128;
VALIDATION_SIZE = 128;
EPOCHE = 50;
ETA = 1;
MOMENTUM = 0.8;
%% Load Data

load('X.mat');
load('Y.mat');

<<<<<<< HEAD
%load('X10K.mat');
%load('Y10K.mat');

XT = X(:,:,1:3968);
YT = Y(1:3968);

XV = X(:,:,3969:3968+1920);
YV = Y(3969:3968+1920);
=======
XT = X(:,:,1:TRAIN_SIZE);
YT = Y(1:TRAIN_SIZE);

XV = X(:,:,TRAIN_SIZE+1:TRAIN_SIZE+1+VALIDATION_SIZE);
YV = Y(TRAIN_SIZE+1:TRAIN_SIZE+1+VALIDATION_SIZE);
>>>>>>> 518a073bbbb31d7406b162a590e9b08185ecd7e1

YT = build_Y(YT);
YV = build_Y(YV);
%% Load layers

load('layers2.mat');
layers2{1}.dim=[28 28 1];
layers2{3}.n_neurons = 10;
layers2{2}.stride=2;
layers2{2}.padding=0;
%% load net
<<<<<<< HEAD
net = net_conv_FC(layers2, {@relu, @sigmoid}, {@reluDeriv, @sigmoidDeriv}, 3);
[err, final_net, err_val] = learningPhase_convFC(net, 20, XT, YT, XV, YV, @softMaxCrossEntropy, @softMaxCrossEntropyDeriv, 2, 0.001, 0.9, 128);
=======
>>>>>>> 518a073bbbb31d7406b162a590e9b08185ecd7e1

net = net_conv_FC(layers2, {@sigmoid, @sigmoid }, {@sigmoidDeriv, @sigmoidDeriv}, 3);
[err, final_net, err_val] = learningPhase_convFC(net, EPOCHE, XT, YT, XV, YV, @softMaxCrossEntropy, @softMaxCrossEntropyDeriv, 2, ETA, MOMENTUM, 128);

%% "GRID SEARCH" per eta e momentum
min_val_err = err_val;
best_net = final_net;
best_eta = 0;
best_momentum = 0;
%{
for eta=0:0.01:1
    for momentum=0:1
        [err, final_net, err_val] = learningPhase_convFC(net, EPOCHE, XT, YT, XV, YV, @softMaxCrossEntropy, @softMaxCrossEntropyDeriv, 2, eta, momentum, 128);
        if(err_val<min_val_err)
            disp(['Best eta ', eta]);
            disp(['Best momentum ', momentum]);
            min_val_err = err_val;
            best_net = final_net;
            best_eta = eta;
            best_momentum = momentum;
        end
    end
end
%}
%TEST
% [a,z] = forward_step_convFC(final_net, X(:,:,1));
% z{2}
% Y(1);
% colormap gray
% imagesc(X(:,:,1));