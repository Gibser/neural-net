%Script per iniziare un training della rete
clear;

%iper-parametri
TRAIN_SIZE = 1024;
VALIDATION_SIZE = 256;
EPOCHE = 50;
ETA = 0.0001;
MOMENTUM = 0.8;
BATCH_SIZE = 32;
TRAINING_TYPE = 2; % 0 (Online), 1 (Batch), 2(Mini-Batch)

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
load('layers.mat');            %cell array per i livelli
load('actvFunc.mat');           %cell array per le funzioni di attivazione
load('actvFuncDeriv.mat');      %cell array per le derivate delle funzioni di attivazione

%% Stampa dei parametri di training
disp('===============Training===================');
disp(['Epoche: ', 9,9, 9, num2str(EPOCHE)]);
disp(['Tipo di Training: ',9 num2str(TRAINING_TYPE)]);
if(TRAINING_TYPE == 2)
disp(['Grandezza batch: ',9 num2str(BATCH_SIZE)]);
end
disp('===============Iper-Parametri==============');
disp(['Training Set: ',9 num2str(TRAIN_SIZE)]);
disp(['Validation Set: ',  num2str(VALIDATION_SIZE)]);
disp(['Learning Rate: ', 9 num2str(ETA)]);
disp(['Momento: ', 9,9 num2str(MOMENTUM)]);
disp('===========================================');

%% Training
net = net_conv_FC(layers2, actvFunc, actvFuncDeriv, 3);
tic
[err, final_net, err_val, acc_tr, acc_val] = learningPhase_convFC(net, EPOCHE, XT, YT, XV, YV, @softMaxCrossEntropy, @softMaxCrossEntropyDeriv, TRAINING_TYPE, ETA, MOMENTUM, BATCH_SIZE);
toc