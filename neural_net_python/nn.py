import numpy as np
from functions import *

class NeuralNetwork():
    def __init__(self, layers, actv, actv_deriv, n_layers):
        SIGMA = 0.1
        self.weights = []
        self.training_weights = []
        self.training_biases = []
        self.biases = []
        self.activations = actv
        self.actv_deriv = actv_deriv
        self.layters = layers
        self.n_layers = n_layers
        for i in range(1, n_layers):
            self.weights.append(SIGMA*np.random.randn(layers[i], layers[i-1]))
            self.biases.append(SIGMA*np.random.randn(layers[i], 1))

    def forward_step(self, x):
        z = x
        a_ = []
        z_ = []
        for i in range(self.n_layers-1):
            if self.training_biases[i].squeeze().ndim != 0:
                a = np.matmul(self.training_weights[i], z) + (self.training_biases[i].squeeze()[:, None])
            else:
                a = np.matmul(self.training_weights[i], z) + (self.training_biases[i].squeeze())
            a_.append(a)
            z = self.activations[i](a)
            print(z)
            z_.append(z)
        return a_, z_

    def backpropagation(self, x, t, derivFunErr):
        W_deriv = []
        deltas = []
        bias_deriv = []
        a_, z_ = self.forward_step(x)
        z_.insert(0, x)

        delta_out = self.actv_deriv[-1](a_[-1])
        delta_out = np.multiply(delta_out, derivFunErr(z_[-1], t))
        deltas.append(delta_out)
        W_deriv.append(np.matmul(delta_out, np.transpose(z_[-1])))

        w = 0
        a = 1
        z = 2
        bias_deriv.append(delta_out)

        for i in range(self.n_layers-2):
            deltas.insert(0, np.matmul(np.transpose(self.weights[-1-w]), deltas[0]))
            print(i)
            deltas[0] = np.multiply(deltas[0], self.actv_deriv[-1-a](a_[-1-a]))
            W_deriv.insert(0, np.matmul(deltas[0], np.transpose(z_[-1-z])))
            bias_deriv.insert(0, deltas[0])
            w += 1
            a += 1
            z += 1

        return W_deriv, bias_deriv

    def gradient_descent(self, eta, W_deriv, bias_deriv):
        for i in range(self.n_layers-1):
            self.training_weights[i] = self.training_weights[i] - eta*W_deriv[i]
            self.training_biases[i] = self.training_biases[i] - eta*bias_deriv[i]

    def train(self, N, x, t, x_val, t_val, errFunc, errFuncDeriv, eta, BATCH):
        err = []
        err_val = []
        self.training_weights = self.weights.copy()
        self.training_biases = self.biases.copy()
        _, z_ = self.forward_step(x_val)
        y_val = z_[1]
        min_err = errFunc(y_val, t_val)

        if BATCH == 1:
            eta = 0.0005

        for epoch in range(N):
            if BATCH == 0:
                for n in range(x.shape[0]):
                    w, b = self.backpropagation(x[:, n], t[:, n], errFuncDeriv)
                    self.gradient_descent(eta, w, b)
            else:
                w, b = self.backpropagation(x, t, errFuncDeriv)
                self.gradient_descent(eta, w, b)

            _, z2_ = self.forward_step(x)
            _, z_ = self.forward_step(x_val)
            y = z2_[-1]
            y_val = z_[-1]
            err.append(errFunc(y, t))
            err_val.append(errFunc(y_val, t_val))
            print('Training error: ' + str(err[epoch]) + ' Validation error: ' + str(err_val[epoch]))

            if err_val[epoch] < min_err:
                min_err = err_val[epoch]
                self.weights = self.training_weights.copy()     #Copio i pesi migliori
                self.biases = self.training_biases.copy()       #Copio i bias migliori
            else:
                self.training_weights = self.weights.copy()
                self.training_biases = self.biases.copy()
        return err, err_val