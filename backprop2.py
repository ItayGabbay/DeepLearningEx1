import numpy as np

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    if activation == "relu":
        dZ = relu_backward(dA, cache[1])
    elif activation == "softmax":
        dZ = softmax_backward(dA, cache[1])

    return linear_backward(dZ, cache[0])


def relu_backward(dA, activation_cache):
    dZ = np.array(dA, copy=True)
    dZ[activation_cache <= 0] = 0

    return dZ


def softmax_backward(dA, activation_cache):
    return dA


def L_model_backward(AL, Y, caches):
    grads = {}
    dAL = (AL - Y) / Y.shape[1]
    num_of_layers = len(caches)
    grads["dA" + str(num_of_layers)], \
    grads["dW" + str(num_of_layers)], \
    grads["db" + str(num_of_layers)] = linear_backward(dAL, caches[num_of_layers-1][0])

    for l in reversed(range(num_of_layers - 1)):
        A_prev, dW, db = linear_activation_backward(grads["dA" + str(l+2)], caches[l], "relu")
        grads["dA" + str(l+1)] = A_prev
        grads["dW" + str(l+1)] = dW
        grads["db" + str(l+1)] = db

    return grads


def update_parameters(parameters, grads, learning_rate):
    num_layers = len(parameters) // 2

    for layer in range(1, num_layers + 1):
        parameters["W" + str(layer)] -= learning_rate * grads["dW" + str(layer)]
        parameters["B" + str(layer)] -= learning_rate * grads["db" + str(layer)]

    return parameters

