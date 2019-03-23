from keras.datasets import mnist
from keras.utils import np_utils
from model import l_layer_model, predict
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
valid_index = np.random.choice(60000, round(60000*0.2), replace=False)
x_valid = x_train[valid_index].T
y_valid = y_train[valid_index]
x_train = np.delete(x_train, valid_index, axis=0).T
y_train = np.delete(y_train, valid_index, axis=0)

layers_dims = [784,20,7,5,10]
network, costs = l_layer_model(x_train, y_train, x_valid, y_valid, layers_dims, 0.009, 3000, 60, False)
predict(x_train, y_train, network)
predict(x_test, y_test, network)
