import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from itertools import count


def batch_data(x, y, batch_size):
    batches = int(np.ceil(x.shape[0] / batch_size))
    batch_x = [x[i*batch_size:i*batch_size + batch_size, ...] for i in range(batches-1)]
    batch_x.append(x[batch_size*(batches-1):, ...])
    batch_y = [y[i*batch_size:i*batch_size + batch_size, ...] for i in range(batches-1)]
    batch_y.append(y[batch_size*(batches-1):, ...])
    return batch_x, batch_y


class SequentialModel:
    def __init__(self):
        self.layers = list()
        self.history = {'loss': list(), 'val_loss': list()}
    def add(self, layers):
        if isinstance(layers, list):
            self.layers += layers
        else:
            self.layers.append(layers)
    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
        for i, layer in enumerate(self.layers[1:], start=1):
            layer.compile(self.layers[i-1].output_size)
    def fit(self, train_x, train_y, epochs, batch_size, val_x=None, val_y=None):
        batch_train_x, batch_train_y = batch_data(train_x, train_y, batch_size)
        if val_x is not None and val_y is not None:
            batch_val_x, batch_val_y = batch_data(val_x, val_y, batch_size)
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            train_loss = list()
            for i, (x, y) in enumerate(zip(batch_train_x, batch_train_y)):
                outputs = self.forward(x)
                train_loss.append(self.loss(outputs[-1], y.T))
                self.backward(x, y, outputs)
            self.history['loss'].append(sum(train_loss))
            if val_x is not None and val_y is not None:
                val_loss = list()
                for x, y in zip(batch_val_x, batch_val_y):
                    y_pred = self.predict(x)
                    val_loss.append(self.loss(y_pred, y.T))
                self.history['val_loss'].append(sum(val_loss))
    def forward(self, x):
        outputs = list()
        inputs = x
        for layer in self.layers:
            inputs = layer(inputs)
            outputs.append(inputs)
        return outputs
    def backward(self, x, y, outputs):
        gradient = self.loss.grad(outputs[-1], y, self.layers[-1].activation)
        for i, layer in zip(count(len(self.layers)-1, -1), self.layers[::-1]):
            inputs = outputs[i-1].T if i != 0 else x
            gradient = layer.backward(gradient, inputs, outputs[i], self.optimizer)
    def predict(self, x):
        inputs = x
        for layer in self.layers:
            if isinstance(layer, Dropout):
                continue
            inputs = layer(inputs)
        return inputs.T
    def evaluate(self, x, y, batch_size):
        batch_x, batch_y = batch_data(x, y, batch_size)
        loss = 0
        for test_x, test_y in zip(batch_x, batch_y):
            y_pred = self.predict(test_x)
            loss += self.loss(y_pred, test_y.T)
        return loss
    def summary(self):
        model = {'name': [], 'weights': [], 'bias': [], 'params': [], 'input_shape': [], 'output_shape': []}
        for layer in self.layers:
            model['name'].append(layer.name)
            try:
                model['weights'].append(layer.w.shape)
                model['bias'].append(layer.b.shape)
                model['params'].append(int(layer.w.size + layer.b.size))
            except AttributeError:
                model['weights'].append(None)
                model['bias'].append(None)
                model['params'].append(0)
            model['input_shape'].append(layer.input_size)
            model['output_shape'].append(layer.output_size)
        model_df = pd.DataFrame(model)
        print(model_df.to_string(index=False))
    def plot_loss(self):
        plt.plot(range(len(self.history['loss'])), self.history['loss'])
        plt.plot(range(len(self.history['val_loss'])), self.history['val_loss'])
        plt.legend(['training loss', 'validation loss'])
        plt.show()


class Model:
    pass

class CategoricalCrossentropy:
    @staticmethod
    def __call__(y_pred, y_true):
        return -np.sum(y_true * np.log10(y_pred))
    @staticmethod
    def grad(y_pred, y_true, activation):
        return y_pred - y_true.T


class MSE:
    @staticmethod
    def __call__(y_pred, y_true):
        return np.sum(np.square(y_true - y_pred)) / y_true.size
    @staticmethod
    def grad(y_pred, y_true, activation):
        return 2 * (y_pred - y_true.T) / y_true.size


class GradientDescent:
    def __init__(self, lr):
        self.lr = lr
    def __call__(self, dw, db, weights, bias):
        weights -= self.lr * dw
        bias -= self.lr * db


class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-07):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    def __call__(self, dw, db, weights, bias):
        pass
        # TODO


class Linear:
    @staticmethod
    def __call__(z):
        return z
    @staticmethod
    def grad(a):
        return 1


class Sigmoid:
    @staticmethod
    def __call__(z):
        return 1 / (1 + np.exp(-z))
    @staticmethod
    def grad(a):
        return a * (1 - a)

        
class Tanh:
    @staticmethod
    def __call__(z):
        return np.tanh(z)
    @staticmethod
    def grad(a):
        return 1 - np.square(Tanh()(a))


class ReLU:
    @staticmethod
    def __call__(z):
        return np.maximum(z, 0)
    @staticmethod
    def grad(a):
        dz = np.ones(a.shape)
        dz[np.where(a < 0)] = 0
        return dz


class LeakyReLU:
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, z):
        a = z
        a[np.where(z < 0)] = self.alpha * a[np.where(z < 0)]
        return a
    def grad(self, a):
        dz = a
        dz[np.where(dz >= 0)] = 1
        dz[np.where(dz < 0)] = self.alpha
        return dz


class Softmax:
    @staticmethod
    def __call__(z):
        z_e = np.exp(z - np.max(z))
        a = z_e / np.sum(z_e)
        return a
    @staticmethod
    def grad(a):
        # TODO
        # a_col = a.reshape(-1, 1)
        # jacobian = np.diagflat(a_col) - np.dot(a_col, a_col.T)
        return 1


class Input:
    def __init__(self, input_size, name=None):
        self.input_size = self.output_size = input_size
        self.name = name if name is not None else 'Input'
    def compile(self, input_size):
        return
    def __call__(self, x):
        return x
    def backward(self, gradient, inputs, outputs, optimizer):
        return gradient


class Flatten:
    def compile(self, input_size):
        self.input_size = input_size
        self.output_size = np.prod(input_size)
        self.name = 'Flatten'
    def __call__(self, x):
        return x.reshape(x.shape[0], -1).T
    def backward(self, gradient, inputs, outputs, optimizer):
        return gradient


class Dense:
    def __init__(self, n, activation, name=None):
        self.n = n
        self.activation = activation
        self.name = name if name is not None else 'Dense ' + str(n)
        self.input_size = None
        self.output_size = self.n
        self.w = None
        self.b = None
    def compile(self, input_size):
        self.input_size = input_size
        self.w = np.random.randn(self.n, input_size) * np.sqrt(2 / (input_size))
        self.b = np.zeros((self.n, 1))
    def __call__(self, x):
        z = np.dot(self.w, x) + self.b
        a = self.activation(z)
        return a
    def backward(self, gradient, inputs, outputs, optimizer):
        gradient *= self.activation.grad(outputs)
        new_gradient = np.dot(self.w.T, gradient)
        dw = np.dot(gradient, inputs.reshape(inputs.shape[0], -1))
        db = np.sum(gradient, axis=1).reshape(-1, 1)
        assert(dw.shape == self.w.shape and db.shape == self.b.shape)
        optimizer(dw, db, self.w, self.b)
        return new_gradient


class Dropout:
    def __init__(self, p, name=None):
        self.p = p
        self.name = name if name is not None else 'Dropout ' + str(p)
    def compile(self, input_size):
        self.input_size = self.output_size = input_size
    def __call__(self, x):
        mask = (np.random.random(x.shape) > self.p) / (1 - self.p)
        return mask * x
    def backward(self, gradient, inputs, outputs, optimizer):
        mask = (output > 0) / (1 - self.p)
        return gradient * mask


class Conv2D:
    def __init__(self, n, ksize, stride=1, padding='same', activation=None, name=None):
        self.n = n
        try:
            self.ksize = tuple(ksize)
        except TypeError:
            self.ksize = (ksize, ksize)
        assert(self.ksize[0] % 2 and self.ksize[1] % 2)
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.name = name if name is not None else 'Conv2D ' + str(n) + 'x' + str(self.ksize)
    def compile(self, input_size):
        self.input_size = input_size
        assert(len(input_size) == 2)
        self.padsize = np.floor(np.divide(self.ksize, 2)).astype(np.int32)
        strided_size = tuple(np.ceil(np.divide(self.input_size, self.stride)).astype(np.int32))
        self.output_size = strided_size if self.padding == 'same' else tuple(np.subtract(strided_size, 2*self.padsize))
        # TODO: correct inicialization
        self.w = np.random.standard_normal((np.product(self.ksize), self.n)) * np.sqrt(2/np.product(self.output_size)/self.stride)
        self.b = np.random.standard_normal((self.n, 1)) * np.sqrt(2/np.product(self.output_size)/self.stride)
    def __call__(self, x):
        x = np.sum(x, axis=-1)
        if self.padding == 'same':
            padded = np.zeros((x.shape[0], ) + tuple(np.add(self.input_size, 2*self.padsize)))
            padded[:, self.padsize[0]:self.input_size[0]+self.padsize[0], self.padsize[1]:self.input_size[1]+self.padsize[1]] = x
            x = padded
        y = np.zeros((x.shape[0], self.output_size[0], self.output_size[1], self.n), dtype=np.float32)
        for i_out, i_in in enumerate(range(0, self.input_size[0], self.stride)):
            for j_out, j_in in enumerate(range(0, self.input_size[1], self.stride)):
                # doing cross-correlation instead of convolution
                area = x[:, i_in:i_in+self.ksize[0], j_in:j_in+self.ksize[1]].reshape(x.shape[0], -1)
                y[:, i_out, j_out, :] = np.dot(area, self.w) + self.b
        return y
    def backward(self, gradient, inputs, outputs, optimizer):
        pass


class MaxPool2D:
    def __init__(self, ksize, name=None):
        try:
            self.ksize = tuple(ksize)
        except TypeError:
            self.ksize = (ksize, ksize)
        self.name = name if name is not None else 'Maxpool ' + str(self.ksize)
    def compile(self, input_size):
        self.input_size = input_size
        assert(len(input_size) == 2)
        self.output_size = tuple(np.ceil(np.divide(input_size, self.ksize)).astype(np.int32))
        self.padding = np.any(np.mod(input_size, 2).astype(np.bool))
    def __call__(self, x):
        if self.padding:
            padded = np.full(x.shape[:1] + tuple(np.multiply(self.output_size, self.ksize)) + x.shape[3:], np.nan)
            padded[:, :self.input_size[0], :self.input_size[1], ...] = x
            x = padded
        shape = (x.shape[0], self.output_size[0], self.ksize[0], self.output_size[1], self.ksize[1]) + x.shape[3:]
        return np.nanmax(x.reshape(shape), axis=(2, 4))
    def backward(self, gradient, inputs, outputs, optimizer):
        pass


class Add:
    def compile(self, input_size):
        self.input_size = self.output_size = input_size
        self.name = 'Add'
    def __call__(self, x):
        return sum(x)
    def backward(self, gradient, inputs, outputs, optimizer):
        pass


class Concatenate:
    def __init__(self, axis=-1, name=None):
        self.axis = axis
        self.name = name if name is not None else 'Concat axis - ' + str(axis)
    def compile(self, input_size):
        self.input_size = self.output_size = input_size
    def __call__(self, x):
        return np.concatenate(x, axis=self.axis)
    def backward(self, gradient, inputs, outputs, optimizer):
        pass
