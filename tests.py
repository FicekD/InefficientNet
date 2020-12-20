import unittest
import numpy as np
import inefficientnet


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


if __name__ == '__main__':
    unittest.main()
