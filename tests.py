import unittest
import numpy as np
import inefficientnet
from scipy import signal


class SingleLayerTests(unittest.TestCase):
    def test_flatten_layer(self):
        layer = inefficientnet.Flatten()
        two_dim_input = np.random.rand(16, 64)
        three_dim_input = np.random.rand(16, 32, 32)
        four_dim_input = np.random.rand(16, 32, 32, 3)
        self.assertEqual(layer(two_dim_input).shape, (64, 16), 'incorrect output shape')
        self.assertEqual(layer(three_dim_input).shape, (32*32, 16), 'incorrect output shape')
        self.assertEqual(layer(four_dim_input).shape, (32*32*3, 16), 'incorrect output shape')

    def test_dense_layer(self):
        layer = inefficientnet.Dense(16, inefficientnet.Linear)
        layer.compile(32)
        self.assertEqual(layer.w.shape, (16, 32), 'incorrect weights shape')
        self.assertEqual(layer.b.shape, (16, 1), 'incorrect bias shape')
        x = np.random.rand(32)
        y = layer(x)
        self.assertTrue(np.all(y == np.dot(layer.w, x) + layer.b), 'incorrect output')

    def test_dropout_layer(self):
        layer = inefficientnet.Dropout(0)
        x = np.random.rand(32)
        self.assertTrue(np.all(x == layer(x)))
        layer = inefficientnet.Dropout(1)
        self.assertTrue(np.all(layer(x) == 0))

    def test_maxpool2d_layer(self):
        layer = inefficientnet.MaxPool2D(2)
        layer.compile((4, 5))
        x = np.zeros((2, 4, 5, 2))
        x[0, 0, 0, 0] = 43
        x[0, 3, 3, 0] = 22
        # y = [43, 0; 0, 22]
        x[1, 0, 0, 0] = 31
        x[1, 0, 1, 0] = 55
        x[1, 1, 0, 0] = 75
        x[1, 1, 1, 0] = 92
        # y = [92, 0; 0, 0]
        x[0, 0, 0, 1] = 1
        x[0, 0, 2, 1] = -25
        x[0, 2, 0, 1] = -7
        x[0, 2, 2, 1] = 55
        # y = [1, 0; 0, 55]
        y = layer(x).astype(np.int32)
        self.assertTrue(np.all(y[0, :, :, 0] == np.array([[43, 0, 0], [0, 22, 0]])))
        self.assertTrue(np.all(y[1, :, :, 0] == np.array([[92, 0, 0], [0, 0, 0]])))
        self.assertTrue(np.all(y[0, :, :, 1] == np.array([[1, 0, 0], [0, 55, 0]])))
        self.assertTrue(np.all(y[1, :, :, 1] == np.array([[0, 0, 0], [0, 0, 0]])))

    def test_conv2d_layer(self):
        layer = inefficientnet.Conv2D(2, 3)
        layer.compile((5, 5))
        x = np.array(list(range(100))).reshape(2, 5, 5, 2)
        layer.kernels[:, :, 0] = np.array(list(range(1, 10))).reshape(3, 3)
        layer.kernels[:, :, 1] = np.array(list(range(-10, -1))).reshape(3, 3)
        y = layer(x)
        x = np.sum(x, -1)
        self.assertTrue(np.all(y[0, :, :, 0] == signal.correlate2d(x[0, :, :], layer.kernels[:, :, 0], 'same')))
        self.assertTrue(np.all(y[0, :, :, 1] == signal.correlate2d(x[0, :, :], layer.kernels[:, :, 1], 'same')))
        self.assertTrue(np.all(y[1, :, :, 0] == signal.correlate2d(x[1, :, :], layer.kernels[:, :, 0], 'same')))
        self.assertTrue(np.all(y[1, :, :, 1] == signal.correlate2d(x[1, :, :], layer.kernels[:, :, 1], 'same')))


if __name__ == '__main__':
    unittest.main()
