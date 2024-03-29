eta=0.001;
MAX_EPOCHES=100000;


X = randn(640, 1)';
T = sin(X);

%% SHUFFLE
ind=randperm(size(X,2));
X=X(:,ind);
T=T(:,ind);

%% SHUFFLE
ind=randperm(size(X,2));
X=X(:,ind);
T=T(:,ind);

XVal=randn(640, 1)';
TVal= sin(XVal);

XTest=randn(600, 1)';
TTest= sin(XTest);

tic
%n = NeuralNetwork([1 50 1], {@sigmoid, @identity}, {@sigmoidDeriv, @identityDeriv}, 3);
%n.train(MAX_EPOCHES, X, T, XVal, TVal, @sumOfSquares, @sumOfSquaresDeriv, eta, 1);
n = net([1 3 1], {@sigmoid, @identity}, {@sigmoidDeriv, @identityDeriv}, 3);
%disp([size(n.weightMomentums), size(n.weights)]);
[err, n, err_val] = learningPhase2(n, MAX_EPOCHES, X, T, XVal, TVal, @sumOfSquares, @sumOfSquaresDeriv, 2, eta, 0.8, 32);
%[err, n, err_val] = learningPhase2(n, MAX_EPOCHES, X, T, XVal, TVal, @MSE, @MSEDeriv, 1);

toc