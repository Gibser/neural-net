import numpy as np


class NeuralNetwork():
    def __init__(self, layers, actv, actv_deriv, n_layers):
        SIGMA = 0.1
        self.weights = np.array()
        self.biases = np.array()
        self.activations = np.array()
        self.actv_deriv = actv_deriv
        self.layters = layers
        self.n_layers = n_layers
        for i in range(2, n_layers+1):
            self.weights[i-1] = SIGMA*np.random.randn(layers[i], layers[i-1])
            self.biases[i-1] = SIGMA*np.random.randn(layers[i], 1)
            self.activations[i-1] = actv[i-1]
            self.actv_deriv[i-1] = actv_deriv[i-1]
