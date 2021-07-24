clear;


TRAIN_SIZE = 128;
VALIDATION_SIZE = 128;
EPOCHE = 100;
ETA = 0.01;
MOMENTUM = 0.8;
%% Load Data

load('X.mat');
load('Y.mat');

XT = X(:,:,1:TRAIN_SIZE);
YT = Y(1:TRAIN_SIZE);

XV = X(:,:,TRAIN_SIZE+1:TRAIN_SIZE+VALIDATION_SIZE);
YV = Y(TRAIN_SIZE+1:TRAIN_SIZE+VALIDATION_SIZE);


YT = build_Y(YT);
YV = build_Y(YV);
%% Load layers

load('layers2.mat');
layers2{1}.dim=[28 28 1];
layers2{3}.n_neurons = 10;
layers2{2}.n_neurons = 128;
layers2{2}.stride=2;
layers2{2}.padding=0;
%% load net

net = net_conv_FC(layers2, {@relu, @identity}, {@reluDeriv, @identityDeriv}, 3);
[err, final_net, err_val] = learningPhase_convFC(net, EPOCHE, XT, YT, XV, YV, @crossentropy, @softMaxCrossEntropyDeriv, 2, ETA, MOMENTUM, 128);

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