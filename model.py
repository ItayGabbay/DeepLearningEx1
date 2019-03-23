import numpy as np
from feedforward import initialize_parameters, L_model_forward, compute_cost
from backprop import L_model_backward, Update_parameters
import random


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False):

    costs = []
    # Initializing
    network = initialize_parameters(layers_dims)
    iteration_costs = []
    for i in range(0, num_iterations):
        # Dividing to mini batches

        train_mini_batch = _get_mini_batch(X, Y, batch_size)
        eval_mini_batch = _get_mini_batch(X, Y, batch_size)

        AL, caches = L_model_forward(train_mini_batch["X"], network, use_batchnorm)
        cost = np.sum(compute_cost(AL, train_mini_batch["Y"]))
        # print(cost)
        iteration_costs.append(cost)

        # Gradient decent
        grads = L_model_backward(AL, train_mini_batch["Y"], caches)

        network = Update_parameters(network, grads, learning_rate)

        total_cost = np.average(iteration_costs)

        if i % 100 == 0:
            costs.append(total_cost)
            print("Iteration:", i, " Total cost is:", total_cost)

        predict(eval_mini_batch["X"], eval_mini_batch["Y"], network)


    return network, costs


def predict(X, Y, parameters):

    m = X.shape[1]
    layers = len(parameters) // 2
    predictions = np.zeros((10, m))

    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        # if probas[0, i] > 0.5:
        predictions[np.argmax(probas[:,i]), i] = 1
        # else:
        #     predictions[0, i] = 0

    print("Accuracy: " + str(np.sum(predictions == Y)/m))

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
#
# def _divide_to_mini_batches(X, Y, batch_size):
#     batches = []
#
#     # It's faster to select from axis 0
#     trans_X = X.T
#     num_of_samples = trans_X.shape[0]
#     shuffled_indexes = list(range(0, num_of_samples))
#     random.shuffle(shuffled_indexes)
#     shuffled_indexes = np.array(shuffled_indexes)
#     for batch_num in range(0, num_of_samples // batch_size):
#         batch_indexes = shuffled_indexes.take(list(range(batch_num*batch_size,(batch_num+1)*batch_size)), axis=0)
#         minibatch = {"X": np.take(trans_X, batch_indexes, axis=0).T, "Y": np.take(Y, batch_indexes, axis=0).T}
#         batches.append(minibatch)
#
#     return batches
