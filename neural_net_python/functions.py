import numpy as np
from scipy.special import expit, logit

def sumOfSquares(y, t):
    return (1/2) * np.sum(np.sum(np.multiply((y-t), 2)))

def sumOfSquaresDeriv(y, t):
    return y-t

def sigmoid(x):
    return expit(x)

def sigmoidDeriv(x):
    z = sigmoid(x)
    return np.multiply(z, 1-z)

def crossEntropyMC(y, t):
    return -np.sum(np.sum(np.multiply(t, np.log(y)), 0))

def crossEntropyMCDeriv(y, t):
    return -np.sum(np.multiply(t, y), 1)

def identity(x):
    return x

def identityDeriv(x):
    return 1