#from keras.datasets import mnist

from nn import NeuralNetwork
from functions import *

eta = 0.0005
MAX_EPOCHS = 200
'''
(X, T), (X_t, T_t) = mnist.load_data()

X_r = X[:500, :, :]
T_r = T[:500]

XTrain = X_r[:200, :, :]
XTrain = XTrain.reshape(XTrain.shape[0], XTrain.shape[1]*XTrain.shape[2])
TTrain = T_r[:200]

XVal = X_r[201:401, :, :]
XVal = XVal.reshape(XVal.shape[0], XVal.shape[1]*XVal.shape[2])
TVal = T_r[201:401]

XTest = X_t[:200, :, :]
XTest = XTest.reshape(XTest.shape[0], XTest.shape[1]*XTest.shape[2])
TTest = T_t[:200]
'''

X = np.random.randn(600)
Y = np.sin(X)

XVal = np.random.randn(100)
YVal = np.sin(XVal)


net = NeuralNetwork([X.shape[0], 50, 1], [sigmoid, identity], [sigmoidDeriv, identityDeriv], 3)
#print(net.weights)
net.train(MAX_EPOCHS, np.transpose(X), Y, np.transpose(XVal), YVal, sumOfSquares, sumOfSquaresDeriv, eta, 1)
