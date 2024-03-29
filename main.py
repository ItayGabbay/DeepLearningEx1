from keras.datasets import mnist
from keras.utils import np_utils
from model import l_layer_model, predict
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, -1) / 255.0
x_test = x_test.reshape(10000, -1) / 255.0
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
network, train_costs, valid_costs = l_layer_model(x_train, y_train, x_valid, y_valid, layers_dims, 0.009, 10000, 600, False, 0.9)
train_pred, train_acc = predict(x_train, y_train.T, network, False)
valid_pred, valid_acc = predict(x_valid, y_valid.T, network, False)
test_pred, test_acc = predict(x_test.T, y_test.T, network, False)

with open('./costs_dropout.txt', 'w') as f:
    f.write("--------TRAIN COSTS--------- \n")
    f.write(str(train_costs))
    f.write("\n--------VALID COSTS---------\n")
    f.write(str(valid_costs))

with open('./results_dropout.txt', 'w') as f:
    f.write("Training Accuracy:" + str(train_acc))
    f.write("\n")
    f.write("Validation Accuracy:" + str(valid_acc))
    f.write("\n")
    f.write("Test Accuracy:" + str(test_acc))
