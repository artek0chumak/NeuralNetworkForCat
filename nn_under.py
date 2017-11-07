import nn_mat
import cudamat as cm
import numpy as np

def initialize_parameters(dim_layer):
    parameters = {}
    for l in range(1, len(dim_layer)):
        parameters.update({'W' + str(l) : np.random.randn(dim_layer[l], dim_layer[l - 1]) * np.sqrt(2 / dim_layer[l - 1])})
        parameters.update({'b' + str(l) : np.zeros((dim_layer[l], 1))})

    return parameters


def linear_forward(A, W, b):
    W1 = cm.CUDAMatrix(W)
    A1 = cm.CUDAMatrix(A)
    b1 = cm.CUDAMatrix(b)
    Z1 = cm.empty((W.shape[0], 1))
    Z1 = cm.dot(W1, A1)
    Z1.add_col_vec(b1)

    Z = Z1.asarray()

    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = nn_mat.sigmoid(Z), Z

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = nn_mat.relu(Z), Z

    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = nn_mat.tanh(Z), Z

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


def compute_cost(AL, Y, parameters, l=0):
    m = Y.shape[1]
    W = list()
    for i in range(len(parameters) // 2):
        W.append(parameters["W" + str(l + 1)])

    W1 = list()
    for i in range(len(W)):
        W1.append(cm.CUDAMatrix(W[i]))

    AL1 = cm.CUDAMatrix(AL)
    Y1 = cm.CUDAMatrix(Y)
    cost1 = cm.empty(AL.shape)
    L2_regularization = 0
    for i in range(len(W1)) :
        L2_regularization += np.squeeze(cm.pow(W1[i], 2).sum(axis=0).sum(axis=1).asarray())


    L2_regularization *= l / m / 2

    cost1 = cm.log(AL1).mult(Y1).add(cm.log(cm.CUDAMatrix(np.ones(AL.shape)).subtract(AL1)).mult(cm.CUDAMatrix(np.ones(Y.shape)).subtract(Y1)))
    cost = np.squeeze(cost1.sum(axis=1).divide(m).asarray()) + L2_regularization

    return -cost.sum()


def linear_backward(dZ, cache, l=0):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    A_prev1 = cm.CUDAMatrix(A_prev)
    W1 = cm.CUDAMatrix(W)
    Wl = cm.CUDAMatrix(W).mult(l)
    b1 = cm.CUDAMatrix(b)
    dA_prev1 = cm.empty(A_prev.shape)
    dW1 = cm.empty(W.shape)
    db1 = cm.empty(b.shape)
    dZ1 = cm.CUDAMatrix(dZ)

    dW1 = cm.dot(dZ1, A_prev1.transpose()).divide(m)
    dW1.add(Wl.divide(m))
    db1 = dZ1.sum(axis = 1).divide(m)
    dA_prev1 = cm.dot(W1.transpose(), dZ1)

    dA_prev = dA_prev1.asarray()
    dW = dW1.asarray()
    db = db1.asarray()

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = nn_mat.der_relu(dA)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = nn_mat.der_sigmoid(dA)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "tanh":
        dZ = nn_mat.der_tanh(dA)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    grads = {}
    Y = Y.reshape(AL.shape)

    dAL = - Y / AL + (1 - Y) / (1 - AL)

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
        W1 = cm.CUDAMatrix(parameters["W" + str(l + 1)])
        b1 = cm.CUDAMatrix(parameters["b" + str(l + 1)])
        dW1 = cm.CUDAMatrix(grads["dW" + str(l + 1)])
        db1 = cm.CUDAMatrix(grads["db" + str(l + 1)])

        W1.subtract(dW1.mult(learning_rate))
        b1.subtract(db1.mult(learning_rate))

        parameters["W" + str(l + 1)] = W1.asarray()
        parameters["b" + str(l + 1)] = b1.asarray()


    return parameters