%Script per iniziare un training della rete
clear;

%iper-parametri
TRAIN_SIZE = 1024;
VALIDATION_SIZE = 512;
EPOCHE = 50;
ETA = 0.0001;
MOMENTUM = 0.8;
BATCH_SIZE = 32;

%% Caricamento dei dati
load('X.mat');                  %Immagini MNIST
load('Y.mat');                  %Labels MNIST

%% Definizione del training set e validation set
XT = X(:,:,1:TRAIN_SIZE);
YT = Y(1:TRAIN_SIZE);

XV = X(:,:,TRAIN_SIZE+1:TRAIN_SIZE+VALIDATION_SIZE);
YV = Y(TRAIN_SIZE+1:TRAIN_SIZE+VALIDATION_SIZE);

%% Costruisce la one-hot encoding per i label degli insiemi
YT = build_Y(YT);
YV = build_Y(YV);


%% Carica i Layers della rete
load('layers2.mat');            %cell array per i livelli
load('actvFunc.mat');           %cell array per le funzioni di attivazione
load('actvFuncDeriv.mat');      %cell array per le derivate delle funzioni di attivazione

%% Training
net = net_conv_FC(layers2, actvFunc, actvFuncDeriv, 3);
tic
[err, final_net, err_val, acc_tr, acc_val] = learningPhase_convFC(net, EPOCHE, XT, YT, XV, YV, @softMaxCrossEntropy, @softMaxCrossEntropyDeriv, 0, ETA, MOMENTUM, BATCH_SIZE);
toc