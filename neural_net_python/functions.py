import numpy as np


def sumOfSquares(y, t):
    return (1/2) * sum(sum(np.multiply((y-t), 2)))

def sumOfSquaresDeriv(y, t):
    return y-t

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDeriv(x):
    z = sigmoid(x)
    return np.multiply(z, 1-z)

def crossEntropyMC(y, t):
    return -sum(sum(np.multiply(t, np.log(y)), 1))

def crossEntropyMCDeriv(y, t):
    return -sum(np.multiply(t, y), 2)