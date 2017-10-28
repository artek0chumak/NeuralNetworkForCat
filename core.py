import numpy as np
import pickle
import matplotlib.pyplot as plt

def sigmoid(X):

    return 1/(1 + np.exp(X))

def load_data_set():
    X_ = list()
    Y_ = list()
    for i in range(5):
        with open('cifar-10-batches-py/data_batch_{0}'.format(i + 1), 'rb') as img:
            data = pickle.load(img, encoding='bytes')
        with open('cifar-10-batches-py/batches.meta'.format(i + 1), 'rb') as lb:
            labels = pickle.load(lb, encoding='bytes')
        for j in range(10000):
            X_.append(data[b'data'][j])
            Y_.append(data[b'labels'][j])

    return np.array(X_).T, np.array(Y_).reshape((1, 50000))

def initialize_parameters(n_x, n_h1, n_h2, n_y):
    W1 = np.random.rand(n_h1, n_x) * 0.0001
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.rand(n_h2, n_h1) * 0.0001
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.rand(n_y, n_h2) * 0.0001
    b3 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h1, n_x))
    assert (b1.shape == (n_h1, 1))
    assert (W2.shape == (n_h2, n_h1))
    assert (b2.shape == (n_h2, 1))
    assert (W3.shape == (n_y, n_h2))
    assert (b3.shape == (n_y, 1))

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3' : W3, 'b3': b3}

    return parameters

def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']


    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0, Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.maximum(0, Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = np.maximum(0,Z3)

    assert (A3.shape == (1, X.shape[1]))

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3' : Z3, 'A3' : A3}

    return A3, cache

def compute_cost(A3, Y, parameters):
    m = Y.shape[1]

    sqr = np.square(Y - A3)
    cost = np.sum(sqr) / m

    cost = np.squeeze(cost)
    assert (isinstance(cost, float))
    return cost

def d_max(x):

    x[x <= 0] = 0
    x[x > 0] = 1

    return x

def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]
    W2 = parameters['W2']
    W3 = parameters['W3']
    A1 = cache['A1']
    A2 = cache['A2']
    A3 = cache['A3']
    Z1 = cache['Z1']
    Z2 = cache['Z2']

    dA3 = 2*(A3 - Y)
    dZ3 = np.multiply(dA3, np.multiply(A3, (1 - A3)))
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, d_max(Z2))
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, d_max(Z1))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

    return grads

def update_parameters(parameters, grads, learning_rate=0.0000002):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    dW3 = grads['dW3']
    db3 = grads['db3']

    W1 = W1 - dW1 * learning_rate
    b1 = b1 - db1 * learning_rate
    W2 = W2 - dW2 * learning_rate
    b2 = b2 - db2 * learning_rate
    W3 = W3 - dW3 * learning_rate
    b3 = b3 - db3 * learning_rate

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

    return parameters

def nn_model(X, Y, n_h1, n_h2, num_iteration=10000, print_cost=False):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    parameters = initialize_parameters(n_x, n_h1, n_h2, n_y)

    for i in range(num_iteration):

        A3, cache = forward_propagation(X, parameters)
        cost = compute_cost(A3, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 10 == 0:
            print("Cost after iteration {0}: {1}".format(i, cost))

    return parameters