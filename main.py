from keras.datasets import mnist
from model import l_layer_model, predict
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).T
x_test = x_test.reshape(10000, 784).T
y_train = y_train.reshape(1, 60000)
y_test = y_test.reshape(1, 10000)
print(x_train.shape)
layers_dims = [784,20,7,5,10]
network, costs = l_layer_model(x_train, y_train, layers_dims, 0.0009, 20, 60, False)
predict(x_train, y_train, network)
predict(x_test, y_test, network)
