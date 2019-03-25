import numpy as np
from feedforward import initialize_parameters, L_model_forward, compute_cost
from backprop2 import L_model_backward, update_parameters
import random


def l_layer_model(X, Y, x_valid, y_valid, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False):

    costs = []
    # Initializing
    network = initialize_parameters(layers_dims)
    iteration_count = 0
    epochs = 0
    not_improving_iterations = 0
    best_val_accuracy = 0
    while not_improving_iterations < 300:
        epochs += 1
        print("Starting Epoch number", epochs)

        # Dividing to mini batches
        mini_batches = _divide_to_mini_batches(X, Y, batch_size)

        for minibatch in mini_batches:
            AL, caches = L_model_forward(minibatch["X"], network, use_batchnorm)
            cost = np.sum(compute_cost(AL, minibatch["Y"]))

            # Gradient decent
            grads = L_model_backward(AL, minibatch["Y"], caches)

            network = update_parameters(network, grads, learning_rate)

            if iteration_count % 100 == 0:
                costs.append(cost)
                predictions, accuracy = predict(x_valid, y_valid.T, network)
                print("Iteration:", iteration_count, " Total cost is:", cost)

            if accuracy  > best_val_accuracy:
                best_val_accuracy = accuracy
                not_improving_iterations = 0
            else:
                not_improving_iterations += 1

            if not_improving_iterations == 300:
                break

            iteration_count += 1


    return network, costs


def predict(X, Y, parameters):

    m = X.shape[1]
    predictions = np.zeros((10, m))

    probas, caches = L_model_forward(X, parameters)
    correct = 0
    for i in range(0, probas.shape[1]):
        predictions[np.argmax(probas[:,i]), i] = 1

        if Y[np.argmax(probas[:,i]), i] == 1:
            correct += 1

    accuracy = correct / m
    print("Accuracy: " + str(accuracy))

    return predictions, accuracy


# def _get_mini_batch(X, Y, batch_size):
#     # It's faster to select from axis 0
#     trans_X = X.T
#     num_of_samples = trans_X.shape[0]
#     shuffled_indexes = list(range(0, num_of_samples))
#     random.shuffle(shuffled_indexes)
#     shuffled_indexes = np.array(shuffled_indexes)
#     batch_indexes = shuffled_indexes.take(list(range(batch_size)), axis=0)
#     minibatch = {"X": np.take(trans_X, batch_indexes, axis=0).T, "Y": np.take(Y, batch_indexes, axis=0).T}
#
#     return minibatch


def _divide_to_mini_batches(X, Y, batch_size):
    batches = []

    # It's faster to select from axis 0
    trans_X = X.T
    num_of_samples = trans_X.shape[0]
    shuffled_indexes = list(range(0, num_of_samples))
    random.shuffle(shuffled_indexes)
    shuffled_indexes = np.array(shuffled_indexes)
    for batch_num in range(0, num_of_samples // batch_size):
        batch_indexes = shuffled_indexes.take(list(range(batch_num*batch_size,(batch_num+1)*batch_size)), axis=0)
        minibatch = {"X": np.take(trans_X, batch_indexes, axis=0).T, "Y": np.take(Y, batch_indexes, axis=0).T}
        batches.append(minibatch)

    return batches