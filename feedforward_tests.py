import unittest
from feedforward import initialize_parameters, linear_forward
import numpy as np


class TestParametersInitialization(unittest.TestCase):
    def test_initialize_parameters(self):
        layer_dims = [784, 20, 7, 5, 10]
        network = initialize_parameters(layer_dims)
        for index, layer in enumerate(layer_dims, 1):
            self.assertEqual(network["W" + str(index)].shape, (layer, layer_dims[index - 1]))

    def test_linear_forward(self):
        network = initialize_parameters([784, 15, 10])
        A = np.random.randn(784, 15)
        z, linear_cache = linear_forward(A=A, W=network["W1"], b=network["B1"])
        self.assertEqual(z.shape, (network["W1"].shape[0], A.shape[1]))


if __name__ == '__main__':
    unittest.main()
