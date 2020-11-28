import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


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
                gradients = self.backward(x, y.T, outputs)
                self.optimizer(self.layers, gradients)
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
        gradients = list()
        for i, layer in zip(range(len(self.layers)-1, -1, -1), self.layers[::-1]):
            try:
                if i == len(self.layers) - 1:
                    da = outputs[-1] - y
                    dz = layer.activation.grad(outputs[-1])
                    grad = da * dz
                else:
                    da = self.layers[i+1].w
                    dz = layer.activation.grad(outputs[i])
                    grad = np.dot(da.T, grad) * dz
                dwl = outputs[i-1].T if i > 1 else x.reshape(x.shape[0], np.product(x.shape[1:]))
                dw = np.dot(grad, dwl)
                db = np.sum(grad, axis=1).reshape(-1, 1)
                assert(dw.shape == layer.w.shape and db.shape == layer.b.shape)
                gradients.append([dw, db])
            except AttributeError:
                gradients.append(None)
        return gradients[::-1]
    def predict(self, x):
        inputs = x
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
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
        print(model_df)
    def plot_loss(self):
        plt.plot(range(len(self.history['loss'])), self.history['loss'])
        plt.plot(range(len(self.history['val_loss'])), self.history['val_loss'])
        plt.legend(['training loss', 'validation loss'])
        plt.show()


class Model:
    pass


def categorical_crossentropy(y_pred, y_true):
    loss = -np.sum(y_true * np.log10(y_pred))
    return loss


class GradientDescent:
    def __init__(self, lr):
        self.lr = lr
    def __call__(self, layers, gradients):
        for layer, gradient in zip(layers, gradients):
            if gradient is None:
                continue
            layer.w -= self.lr * gradient[0]
            layer.b -= self.lr * gradient[1]


class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-07):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    def __call__(self, layers, gradients):
        for layer, gradient in zip(layers, gradients):
            if gradient is None:
                continue
            # TODO


class Sigmoid:
    @staticmethod
    def output(z):
        return 1 / (1 + np.exp(-z))
    @staticmethod
    def grad(a):
        return a * (1 - a)


class ReLU:
    @staticmethod
    def output(z):
        return mp.maximum(z, 0)
    @staticmethod
    def grad(a):
        dz = np.ones(a.shape)
        dz[np.where(a <= 0)] = 0
        return dz


class LeakyReLU:
    def __init__(self, alpha):
        self.alpha = alpha
    def output(self, z):
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
    def output(z):
        z_e = np.exp(z - np.max(z))
        a = z_e / np.sum(z_e)
        return a
    @staticmethod
    def grad(a):
        # TODO
        # a_col = a.reshape(-1, 1)
        # jacobian = np.diagflat(a_col) - np.dot(a_col, a_col.T)
        return a


class Input:
    def __init__(self, input_size, name=None):
        self.input_size = self.output_size = input_size
        self.name = name if name is not None else 'Input'
    def __call__(self, x):
        return x


class Flatten:
    def compile(self, input_size):
        self.input_size = self.output_size = input_size
    def __call__(self, x):
        return x.reshape(-1, x.shape[0])


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
        inputs = np.prod(input_size)
        self.w = np.random.randn(self.n, inputs) * np.sqrt(2 / (inputs))
        self.b = np.random.randn(self.n, 1) * np.sqrt(2 / (inputs))
    def __call__(self, x):
        z = np.dot(self.w, x) + self.b
        a = self.activation.output(z)
        return a


class Dropout:
    def __init__(self, p):
        self.p = p
    def compile(self, input_size):
        self.input_size = self.output_size = input_size
    def __call__(self, x):
        mask = np.random.random(x.shape) > self.p
        return mask * x


class Conv2D:
    def __init__(self, n, ksize, stride=1, padding='same', activation=None):
        self.n = n
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.activation = activation
    def compile(self, input_size):
        self.input_size = input_size
        self.output_size = input_size if self.padding == 'same' else tuple(np.array(input_size) - 2*np.floor(ksize / 2))
    def __call__(self, x):
        # TODO
        pass


class MaxPool2D:
    def __init__(self, ksize, stride=2):
        self.ksize = ksize
        self.stride = stride
    def compile(self, input_size):
        self.input_size = input_size
        self.output_size = tuple(np.ceil(np.array(input_size) / self.stride))
    def __call__(self, x):
        # TODO
        pass


class Add:
    def compile(self, input_size):
        self.input_size = self.output_size = input_size
    def __call__(self, x):
        return sum(x)
