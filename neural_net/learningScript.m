%%%IPERPARAMETRI
M=50;
eta=0.0005;
MAX_EPOCHES=300;
%%%
if not(exist('X'))
    X=loadMNISTImages('mnist/train-images-idx3-ubyte');
    Labels=loadMNISTLabels('mnist/train-labels-idx1-ubyte');
%PROBLEMA A 10 CLASSI

T=getTargetsFromLabels(Labels);
%% SHUFFLE
ind=randperm(size(X,2));
X=X(:,ind);
T=T(:,ind);
end

%% PER QUESTIONI DI "VELOCITA" RIDUCO IL DATASET
X_r=X(:,1:500);
T_r=T(:,1:500);
%% SHUFFLE
ind=randperm(size(X_r,2));
X_r=X_r(:,ind);
T_r=T_r(:,ind);

%% DEVO DIVIDERE ALMENO IL MIO DATASET IN TRAINING, VALIDATION E TEST
% IL VALIDATION MI SERVE OGNI QUALVOLTA C'E' UN PROCESSO DI LEARNING ITERATIVO
% TRAINING, VALIDATION E TEST SET DEVONO ESSERE DEI CAMPIONI RAPPRESENTATIVI DEL PROBLEMA

XTrain= X_r(:,1:200);
TTrain= T_r(:,1:200);

XVal=X_r(:,201:400);
TVal= T_r(:,201:400);

XTest=X_r(:,401:end);
TTest= T_r(:,401:end);
%%
%n = net([size(XTrain,1) 10 10 4 size(TTrain,1)], {@relu, @relu, @relu, @sigmoid}, {@reluDeriv, @reluDeriv, @reluDeriv, @sigmoidDeriv}, 5);
%n2 = net([size(XTrain,1) 10 10 4 size(TTrain,1)], {@sigmoid, @sigmoid, @sigmoid, @sigmoid}, {@sigmoidDeriv, @sigmoidDeriv, @sigmoidDeriv, @sigmoidDeriv}, 5);
n = net([size(XTrain, 1) 50 size(TTrain, 1)], {@sigmoid, @sigmoid}, {@sigmoidDeriv, @sigmoidDeriv}, 3);
[err, new_net, err_val] = learningPhase(n, MAX_EPOCHES, XTrain, TTrain, XVal, TVal, @crossEntropyMCDeriv, eta, 1);
%[err2, new_net2, err_val2] = learningPhase(n2, MAX_EPOCHES, XTrain, TTrain, XVal, TVal, @crossEntropyMCDeriv, eta, 1);
