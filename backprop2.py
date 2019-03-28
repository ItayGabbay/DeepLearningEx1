import numpy as np


def batchnorm_backward(dZ, bn_cache):
    Z, Znorm, mu, var = bn_cache

    N, D = dZ.shape

    Z_mu = Z - mu
    std_inv = 1. / np.sqrt(var + 1e-8)

    dvar = np.sum(dZ * Z_mu, axis=0) * -.5 * std_inv ** 3
    dmu = np.sum(dZ * -std_inv, axis=0) + dvar * np.mean(-2. * Z_mu, axis=0)

    dZ_BN = (dZ * std_inv) + (dvar * 2 * Z_mu / N) + (dmu / N)

    return dZ_BN


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, use_batchnorm):
    if activation == "relu":
        dZ = relu_backward(dA, cache[1])
    elif activation == "softmax":
        dZ = softmax_backward(dA, cache[1])

    if (use_batchnorm):
        dZ = batchnorm_backward(dZ, cache[2])

    return linear_backward(dZ, cache[0])


def relu_backward(dA, activation_cache):
    relu_grad = activation_cache > 0
    dZ = dA * relu_grad

    return dZ


def softmax_backward(dA, activation_cache):
    return dA


def L_model_backward(AL, Y, caches, use_batchnorm=0):
    grads = {}
    # dAL = np.multiply(AL, np.multiply(1 - AL, Y - AL))
    dAL = (AL - Y)
    num_of_layers = len(caches)
    grads["dA" + str(num_of_layers)], \
    grads["dW" + str(num_of_layers)], \
    grads["db" + str(num_of_layers)] = linear_backward(dAL, caches[num_of_layers - 1][0])

    for l in reversed(range(num_of_layers - 1)):
        A_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 2)], caches[l], "relu",
                                                    use_batchnorm=use_batchnorm)
        grads["dA" + str(l + 1)] = A_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

    return grads


def update_parameters(parameters, grads, learning_rate):
    num_layers = len(parameters) // 2

    for layer in range(1, num_layers + 1):
        parameters["W" + str(layer)] -= learning_rate * grads["dW" + str(layer)]
        parameters["B" + str(layer)] -= learning_rate * grads["db" + str(layer)]

    return parameters
