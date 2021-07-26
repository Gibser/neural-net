clear;

TRAIN_SIZE = 2048;
VALIDATION_SIZE = 2048;
EPOCHE = 50;
ETA = 0.0001;
MOMENTUM = 0.8;
BATCH_SIZE = 32;

%% Load Data
load('X.mat');                  %Immagini MNIST
load('Y.mat');                  %Labels MNIST

XT = X(:,:,1:TRAIN_SIZE);
YT = Y(1:TRAIN_SIZE);

XV = X(:,:,TRAIN_SIZE+1:TRAIN_SIZE+VALIDATION_SIZE);
YV = Y(TRAIN_SIZE+1:TRAIN_SIZE+VALIDATION_SIZE);


YT = build_Y(YT);
YV = build_Y(YV);

%% Load layers
load('layers2.mat');            %cell array per i livelli
load('actvFunc.mat');           %cell array per le funzioni di attivazione
load('actvFuncDeriv.mat');      %cell array per le derivate delle funzioni di attivazione

%% Training
net = net_conv_FC(layers2, actvFunc, actvFuncDeriv, 3);
tic
[err, final_net, err_val, acc_tr, acc_val] = learningPhase_convFC(net, EPOCHE, XT, YT, XV, YV, @softMaxCrossEntropy, @softMaxCrossEntropyDeriv, 2, ETA, MOMENTUM, BATCH_SIZE);
toc