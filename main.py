from keras.datasets import mnist
from keras.utils import np_utils
from model import l_layer_model, predict

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).T
x_test = x_test.reshape(10000, 784).T
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

layers_dims = [784,20,7,5,10]
network, costs = l_layer_model(x_train, y_train, layers_dims, 0.009, 3000, 60, False)
predict(x_train, y_train, network)
predict(x_test, y_test, network)
