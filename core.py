import numpy as np
import pickle
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1/(1 + np.exp(X))


def relu(X):
    return np.maximum(X, 0)


def tanh(X):
    return np.tanh(X)


def der_sigmoid(X):
    return np.multiply(sigmoid(X), (1 - sigmoid(X)))


def der_relu(X):
    X[X <= 0] = 0
    X[X > 0] = 1
    return X


def der_tanh(X):
    return 1 / np.square(np.cosh(X))


def load_train_set():
    X_ = list()
    Y_ = list()
    for i in range(5):
        with open('cifar-10-batches-py/data_batch_{0}'.format(i + 1), 'rb') as img:
            data = pickle.load(img, encoding='bytes')
        for j in range(10000):
            X_.append(data[b'data'][j])
            if(data[b'labels'][j] == '2' or data[b'labels'][j] == '3' or data[b'labels'][j] == '5' or data[b'labels'][j] == '6') :
                Y_.append(1)
            else:
                Y_.append(0)

    return np.array(X_).T, np.array(Y_).reshape((1, 50000))


def load_test_set():
    X_ = list()
    Y_ = list()
    with open('cifar-10-batches-py/test_batch') as img:
        data = pickle.load(img, encoding='bytes')
    for i in range(10000):
        X_.append(data[b'data'][i])
        if (data[b'labels'][i] == '2' or data[b'labels'][i] == '3' or data[b'labels'][i] == '5' or data[b'labels'][i] == '6'):
            Y_.append(1)
        else:
            Y_.append(0)

    return np.array(X_).T, np.array(Y_).reshape((1, 10000))


def initialize_parameters(dim_layer):
    parameters = {}
    for l in range(1, len(dim_layer)):
        parameters.update({'W' + str(l) : np.random.randn(dim_layer[l], dim_layer[l - 1]) * np.sqrt(2 / dim_layer[l - 1])})
        parameters.update({'b' + str(l) : np.zeros((dim_layer[l], 1))})

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z), Z

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z), Z

    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z), Z

    cache = (linear_cache, activation_cache)

    return A, cache


def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y, parameters):
    m = Y.shape[1]

    precost = np.square(Y - AL)
    cost = np.sum(precost) / m

    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = der_relu(dA)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = der_sigmoid(dA)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "tanh":
        dZ = der_tanh(dA)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    grads = {}
    Y = Y.reshape(AL.shape)

    dAL = 2 * (AL - Y)

    current_cache = caches[len(caches) - 1]
    grads["dA" + str(len(caches))], grads["dW" + str(len(caches))], grads["db" + str(len(caches))] \
        = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    for l in reversed(range(len(caches) - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 2)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp


    return grads


def update_parameters(parameters, grads, learning_rate=0.0000002):
    for l in range(len(parameters) // 2):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def nn_model(X, Y, layers_dims, num_iteration=10000, print_cost=False, learning_rate=0.002):
    parameters = initialize_parameters(layers_dims)

    for i in range(num_iteration):
        AL, caches = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y, parameters)
        grads = backward_propagation(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 3 == 0:
            print("Cost after iteration {0}: {1}".format(i, cost))

    return parameters