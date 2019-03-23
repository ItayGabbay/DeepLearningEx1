import numpy as np
import sys


def initialize_parameters(layer_dims):
    """

    :param layer_dims: an array of the dimensions of each layer in the network (layer 0 is the size of the
            flattened input, layer L is the output sigmoid)
    :return: network: a dictionary containing the initialized W and b parameters of each layer
            (W1…WL, b1…bL).
    """

    if len(layer_dims) < 2:
        raise NeuralNetworkMustHaveInputAndOutputLayers
    network = {}

    for index in range(1, len(layer_dims)):
        network['W' + str(index)] = np.random.randn(layer_dims[index], layer_dims[index-1]) * 0.01
        network['B' + str(index)] = np.zeros((layer_dims[index], 1))

    return network


def linear_forward(A, W, b):
    """

    :param A: The activations of the previous layer
    :param W: The weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    :param b: The bias vector of the current layer (of shape [size of current layer, 1])

    :return:
        Z: The linear component of the activation function (i.e., the value before applying the non-linear function)
        linear_cache: a dictionary containing A, W, b (stored for making the backpropagation easier to compute)

    """

    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)

    return Z, linear_cache


def linear_activation_forward(A_prev, W, B, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    :param A_prev: Activations of the previous layer
    :param W: The weights matrix of the current layer
    :param B: The bias vector of the current layer
    :param activation: The activation function to be used (a string, either “sigmoid” or “relu”)

    :return:
        A – The activations of the current layer
        cache – A joint dictionary containing both linear_cache and activation_cache

    """

    z, linear_cache = linear_forward(A_prev, W, B)

    if activation == "softmax":
        A, activation_cache = softmax(z)
    elif activation == "relu":
        A, activation_cache = relu(z)
    else:
        raise ActivationFunctionNotFound

    return A, (linear_cache, activation_cache)


def L_model_forward(X, parameters, use_batchnorm=0):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    :param X: The data, numpy array of shape (input size, number of examples)
    :param parameters: The initialized W and b parameters of each layer
    :param use_batchnorm: A boolean flag used to determine whether to apply batchnorm after the activation
    :return:
        AL – The last post-activation value
       caches – A list of all the cache objects generated by the linear_forward function
    """

    caches = list()

    num_of_layers = len(parameters) // 2
    current_A = X
    for layer in range(1, num_of_layers):
        W = parameters["W" + str(layer)]
        B = parameters["B" + str(layer)]
        A, cache = linear_activation_forward(current_A, W, B, activation="relu")
        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(cache)
        current_A = A

    W = parameters["W" + str(num_of_layers)]
    B = parameters["B" + str(num_of_layers)]
    AL, cache = linear_activation_forward(A, W, B, activation="softmax")
    caches.append(cache)

    return AL, caches

def softmax(x):
    A = np.exp(x) / np.sum(np.exp(x), axis=0)
    Z = x
    return A, Z

def compute_cost(AL, Y):
    """

    :param AL: Probability vector corresponding to your label predictions, shape (number_of_classes, number of examples)
    :param Y: the labels vector (i.e. the ground truth)
    :return:
        cost – the cross-entropy cost
    """
    m = Y.shape[1]
    cls_sum = np.zeros(10)

    for sample in range(m):
        for cls in range(Y.shape[0]):
            cls_sum[cls] += Y[cls][sample]*np.log(AL[cls][sample])

    cost = (- cls_sum / m).T
    return cost


def apply_batchnorm(A):
    """
    performs batchnorm on the received activation values of a given layer.

    :param A: The activation values of a given layer
    :return:
        NA - the normalized activation values, based on the formula learned in class

    """
    mean = np.mean(A, axis=0)
    variance = np.var(A, axis=0)

    NA = (A - mean) / np.sqrt(variance + sys.float_info.epsilon)

    return NA


# def sigmoid(Z):
#     """
#
#     :param Z: The linear component of the activation function
#     :return:
#         A – the activations of the layer
#         activation_cache – returns Z, which will be useful for the backpropagation
#
#     """
#     A = 1 / (1 + np.exp(-Z))
#     activation_cache = Z
#     return A, activation_cache


def relu(Z):
    """

    :param Z: The linear component of the activation function

    :return:
        A – the activations of the layer
        activation_cache – returns Z, which will be useful for the backpropagation
    """
    A = np.maximum(Z, 0)
    activation_cache = Z
    return A, activation_cache


class NeuralNetworkMustHaveInputAndOutputLayers(Exception):
    pass


class ActivationFunctionNotFound(Exception):
    pass
