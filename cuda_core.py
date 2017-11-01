import core
import cudamat as cm
import numpy as np
import pickle

def sigmoid(X) :
    X1 = cm.CUDAMatrix(X)
    R1 = cm.CUDAMatrix(np.ones(X.shape))
    R = cm.empty(X.shape)

    cm.exp(X1, R)
    R.add(R1)
    R1.divide(R)

    return R1.asarray()


def relu(X) :
    X1 = cm.CUDAMatrix(X)
    R0 = cm.empty(X.shape)
    R = cm.empty(X.shape)

    R0.maximum(X1, R)

    return R.asarray()


def tanh(X) :
    X1 = cm.CUDAMatrix(X)
    R = cm.empty(X.shape)

    cm.tanh(X1, R)

    return R.asarray()


def der_sigmoid(X) :
    X1 = cm.CUDAMatrix(sigmoid(X))
    R = cm.CUDAMatrix(sigmoid(X))
    R1 = cm.CUDAMatrix(np.ones(X.shape))

    R1.subtract(R)
    X1.mult(R1)

    return X1.asarray()


def der_relu(X) :
    X[X <= 0] = 0
    X[X > 0] = 1
    return X


def der_tanh(X) :
    X1 = cm.CUDAMatrix(X)
    R1 = cm.CUDAMatrix(np.ones(X.shape))
    Ex = cm.empty(X.shape)
    EX = cm.empty(X.shape)

    Ex = cm.exp(X)
    EX = cm.exp(cm.empty(X.shape).subtract(X1))

    R1 = Ex.add(EX).mult(0.5)

    return R1.asarray()


def load_train_set():
    X_ = list()
    Y_ = list()
    for i in range(5):
        with open('cifar-10-batches-py/data_batch_{0}'.format(i + 1), 'rb') as img:
            data = pickle.load(img, encoding='bytes')
        for j in range(10000):
            X_.append(data[b'data'][j])
            Y_.append(data[b'labels'][j] / 11.0)

    return np.array(X_).T, np.array(Y_).reshape((1, 50000))


def load_test_set():
    X_ = list()
    Y_ = list()
    with open('cifar-10-batches-py/test_batch') as img:
        data = pickle.load(img, encoding='bytes')
    for i in range(10000):
        X_.append(data[b'data'][i])
        Y_.append(data[b'labels'][i] / 11.0)

    return np.array(X_).T, np.array(Y_).reshape((1, 10000))


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

    AL1 = cm.CUDAMatrix(AL)
    Y1 = cm.CUDAMatrix(Y)
    cost1 = cm.empty(AL.shape)

    cost1 = cm.pow(AL1.subtract(Y1), 2)
    cost = cost1.sum(axis = 1).divide(m).asarray()

    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    A_prev1 = cm.CUDAMatrix(A_prev)
    W1 = cm.CUDAMatrix(W)
    b1 = cm.CUDAMatrix(b)
    dA_prev1 = cm.empty(A_prev.shape)
    dW1 = cm.empty(W.shape)
    db1 = cm.empty(b.shape)
    dZ1 = cm.CUDAMatrix(dZ)

    dW1 = cm.dot(dZ1, A_prev1.transpose()).divide(m)
    db1 = dZ1.sum(axis = 1).divide(m)
    dA_prev1 = cm.dot(W1.transpose(), dZ1)

    dA_prev = dA_prev1.asarray()
    dW = dW1.asarray()
    db = db1.asarray()

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
        # parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        # parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        W1 = cm.CUDAMatrix(parameters["W" + str(l + 1)])
        b1 = cm.CUDAMatrix(parameters["b" + str(l + 1)])
        dW1 = cm.CUDAMatrix(grads["dW" + str(l + 1)])
        db1 = cm.CUDAMatrix(grads["db" + str(l + 1)])

        W1.subtract(dW1.mult(learning_rate))
        b1.subtract(db1.mult(learning_rate))

        parameters["W" + str(l + 1)] = W1.asarray()
        parameters["b" + str(l + 1)] = b1.asarray()


    return parameters


def nn_model(X, Y, layers_dims, num_iteration=10000, print_cost=False, learning_rate=0.002):
    cm.cuda_set_device(0)
    cm.init()

    parameters = initialize_parameters(layers_dims)

    for i in range(num_iteration):
        AL, caches = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y, parameters)
        grads = backward_propagation(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {0}: {1}".format(i, cost))

    return parameters

    cm.shutdown()
