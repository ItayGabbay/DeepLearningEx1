from keras.datasets import mnist
from model import l_layer_model, predict
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
layers_dims = [784,20,7,5,10]
l_layer_model(x_train, y_train, layers_dims, 0.009, 3000, 60, False)
