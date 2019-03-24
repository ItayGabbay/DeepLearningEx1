import numpy as np
from feedforward import initialize_parameters, L_model_forward, compute_cost
from backprop2 import L_model_backward, update_parameters
import random


def l_layer_model(X, Y, x_valid, y_valid, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False):

    costs = []
    # Initializing
    network = initialize_parameters(layers_dims)
    train_mini_batch = _get_mini_batch(X, Y, batch_size)
    for i in range(0, num_iterations):
        # Dividing to mini batches

        AL, caches = L_model_forward(train_mini_batch["X"], network, use_batchnorm)
        cost = np.sum(compute_cost(AL, train_mini_batch["Y"]))

        # Gradient decent
        grads = L_model_backward(AL, train_mini_batch["Y"], caches)

        network = update_parameters(network, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            print("Iteration:", i, " Total cost is:", cost)
            predict(x_valid, y_valid.T, network)


    return network, costs


def predict(X, Y, parameters):

    m = X.shape[1]
    layers = len(parameters) // 2
    predictions = np.zeros((10, m))

    probas, caches = L_model_forward(X, parameters)
    correct = 0
    for i in range(0, probas.shape[1]):
        predictions[np.argmax(probas[:,i]), i] = 1

        if Y[np.argmax(probas[:,i]), i] == 1:
            correct += 1

    print("Accuracy: " + str(correct / m))

    return predictions


def _get_mini_batch(X, Y, batch_size):
    # It's faster to select from axis 0
    trans_X = X.T
    num_of_samples = trans_X.shape[0]
    shuffled_indexes = list(range(0, num_of_samples))
    random.shuffle(shuffled_indexes)
    shuffled_indexes = np.array(shuffled_indexes)
    batch_indexes = shuffled_indexes.take(list(range(batch_size)), axis=0)
    minibatch = {"X": np.take(trans_X, batch_indexes, axis=0).T, "Y": np.take(Y, batch_indexes, axis=0).T}

    return minibatch

