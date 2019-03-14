import numpy as np
from feedforward import initialize_parameters, L_model_forward, compute_cost
from backprop import L_model_backward, Update_parameters
import random


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False):

    costs = []
    # Initializing
    network = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        # Dividing to mini batches
        mini_batches = _divide_to_mini_batches(X, Y, batch_size)
        iteration_costs = []

        for minibatch in mini_batches:
            AL, caches = L_model_forward(minibatch["X"], network, use_batchnorm)
            cost = compute_cost(AL, minibatch["Y"])
            iteration_costs.append(cost)

            # Gradient decent
            grads = L_model_backward(AL, minibatch["Y"], caches)

            network = Update_parameters(network, grads, learning_rate)

        total_cost = np.average(iteration_costs)

        if i % 100 == 0:
            costs.append(total_cost)

        print("Iteration:", i, " Total cost is:", total_cost)

    return network, costs


def predict(X, Y, parameters):
    raise NotImplementedError


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
        minibatch = {"X": np.take(trans_X, batch_indexes, axis=0).T, "Y": np.take(Y, batch_indexes)}
        batches.append(minibatch)

    return batches
