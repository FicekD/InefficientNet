import unittest
import numpy as np
import inefficientnet
from scipy.signal import correlate


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
        layer = inefficientnet.Dense(16, inefficientnet.Linear())
        layer.compile(32)
        self.assertEqual(layer.w.shape, (16, 32), 'incorrect weights shape')
        self.assertEqual(layer.b.shape, (16, 1), 'incorrect bias shape')
        x = np.random.rand(32)
        y = layer(x)
        self.assertTrue(np.all(y == np.dot(layer.w, x) + layer.b), 'incorrect output')

    def test_dropout_layer(self):
        layer = inefficientnet.Dropout(0)
        x = np.random.rand(32)
        self.assertTrue(np.all(x == layer(x)), 'expected all inputs to pass')
        layer = inefficientnet.Dropout(1)
        self.assertTrue(np.all(layer(x) == 0), 'expected all zeros output')

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
        self.assertTrue(np.all(y[0, :, :, 0] == np.array([[43, 0, 0], [0, 22, 0]])), 'incorrect result')
        self.assertTrue(np.all(y[1, :, :, 0] == np.array([[92, 0, 0], [0, 0, 0]])), 'incorrect result')
        self.assertTrue(np.all(y[0, :, :, 1] == np.array([[1, 0, 0], [0, 55, 0]])), 'incorrect result')
        self.assertTrue(np.all(y[1, :, :, 1] == np.array([[0, 0, 0], [0, 0, 0]])), 'incorrect result')

    def test_conv2d_layer_stride_1(self):
        layer = inefficientnet.Conv2D(2, 3)
        layer.compile((5, 5))
        x = np.arange(100).reshape(2, 5, 5, 2)
        layer.w[:, :] = np.arange(18).reshape(9, 2)
        layer.b = 0
        y = layer(x)
        x = np.sum(x, -1)
        y_true = np.zeros(x.shape + (2, ))
        y_true[:, :, :, 0] = correlate(x, layer.w[:, 0].reshape(1, 3, 3), 'same')
        y_true[:, :, :, 1] = correlate(x, layer.w[:, 1].reshape(1, 3, 3), 'same')
        self.assertTrue(np.all(y == y_true), 'incorrect result for stride == 1')

    def test_conv2d_layer_stride_2(self):
        layer = inefficientnet.Conv2D(2, 3, 2)
        layer.compile((5, 5))
        x = np.arange(100).reshape(2, 5, 5, 2)
        layer.w[:, :] = np.arange(18).reshape(9, 2)
        layer.b = 0
        y = layer(x)
        x = np.sum(x, -1)
        y_true = np.zeros(x.shape[:1] + tuple(np.ceil(np.divide(x.shape[1:], 2)).astype(np.int32)) + (2, ))
        y_true[:, :, :, 0] = correlate(x, layer.w[:, 0].reshape(1, 3, 3), 'same')[:, ::2, ::2]
        y_true[:, :, :, 1] = correlate(x, layer.w[:, 1].reshape(1, 3, 3), 'same')[:, ::2, ::2]
        self.assertTrue(np.all(y == y_true), 'incorrect result for stride == 2')

        layer = inefficientnet.Conv2D(2, 3, 2)
        layer.compile((6, 6))
        x = np.arange(144).reshape(2, 6, 6, 2)
        layer.w[:, :] = np.arange(18).reshape(9, 2)
        layer.b = np.zeros_like(layer.b)
        y = layer(x)
        x = np.sum(x, -1)
        y_true = np.zeros(x.shape[:1] + tuple(np.ceil(np.divide(x.shape[1:], 2)).astype(np.int32)) + (2, ))
        y_true[:, :, :, 0] = correlate(x, layer.w[:, 0].reshape(1, 3, 3), 'same')[:, ::2, ::2]
        y_true[:, :, :, 1] = correlate(x, layer.w[:, 1].reshape(1, 3, 3), 'same')[:, ::2, ::2]
        self.assertTrue(np.all(y == y_true), 'incorrect result for stride == 2')


if __name__ == '__main__':
    unittest.main()
