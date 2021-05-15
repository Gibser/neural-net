%%%IPERPARAMETRI
M=50;
eta=0.00005;
MAX_EPOCHES=1000;
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

XTrain= X_r(:,1:200);
TTrain= T_r(:,1:200);

XVal=X_r(:,201:400);
TVal= T_r(:,201:400);

XTest=X_r(:,401:end);
TTest= T_r(:,401:end);

n = net([size(XTrain, 1) 50 size(TTrain, 1)], {@sigmoid, @sigmoid}, {@sigmoidDeriv, @sigmoidDeriv}, 3);
[err, new_net, err_val] = learningPhase(n, MAX_EPOCHES, XTrain, TTrain, XVal, TVal, @crossEntropyMCDeriv, eta, 1);
