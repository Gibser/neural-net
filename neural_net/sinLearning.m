M=50;
eta=0.0001;
MAX_EPOCHES=10000;


X = randn(5000, 1)';
T = sin(X);

%% SHUFFLE
ind=randperm(size(X,2));
X=X(:,ind);
T=T(:,ind);

%% SHUFFLE
ind=randperm(size(X,2));
X=X(:,ind);
T=T(:,ind);

XVal=randn(1000, 1)';
TVal= sin(XVal);

XTest=randn(100, 1)';
TTest= sin(XTest);

%n = NeuralNetwork([1 50 1], {@sigmoid, @identity}, {@sigmoidDeriv, @identityDeriv}, 3);
%n.train(MAX_EPOCHES, X, T, XVal, TVal, @sumOfSquares, @sumOfSquaresDeriv, eta, 1);
n = net([1 50 1], {@sigmoid, @identity}, {@sigmoidDeriv, @identityDeriv}, 3)
learningPhase(n, MAX_EPOCHES, X, T, XVal, TVal, @sumOfSquares, @sumOfSquaresDeriv, eta, 1);
