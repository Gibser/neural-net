M=50;
eta=0.0001;
MAX_EPOCHES=100000;


X = randn(500, 1)';
T = sin(X);

%% SHUFFLE
ind=randperm(size(X,2));
X=X(:,ind);
T=T(:,ind);

%% SHUFFLE
ind=randperm(size(X,2));
X=X(:,ind);
T=T(:,ind);

XVal=randn(100, 1)';
TVal= sin(XVal);

XTest=randn(100, 1)';
TTest= sin(XTest);

n = net([1 50 1], {@sigmoid, @identity}, {@sigmoidDeriv, @identityDeriv}, 3);
[err, new_net, err_val] = learningPhase(n, MAX_EPOCHES, X, T, XVal, TVal, @sumOfSquares, @sumOfSquaresDeriv, eta, 1);
