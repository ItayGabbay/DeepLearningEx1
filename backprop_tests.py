import unittest
import cv2
from backprop import Linear_backward
from feedforward import L_model_forward, initialize_parameters


class TestBackPropagation(unittest.TestCase):
    def test_Linear_backward(self):
        network = initialize_parameters([728, 15, 10])
        fiveImg = cv2.imread("mnist_first_digit.png")
        AL, caches = L_model_forward(fiveImg, network)
        # dA_prev, dW, db = Linear_backward(dZ, cache)


if __name__ == '__main__':
    unittest.main()
