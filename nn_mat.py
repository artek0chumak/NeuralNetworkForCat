import cudamat as cm
import numpy as np

def sigmoid(X) :
    X1 = cm.CUDAMatrix(X)

    return cm.sigmoid(X1).asarray()


def relu(X) :
    X1 = cm.CUDAMatrix(X)
    R0 = cm.empty(X.shape)
    R = cm.empty(X.shape)

    R0.maximum(X1, R)

    return R0.asarray()


def tanh(X) :
    X1 = cm.CUDAMatrix(X)

    return cm.tanh(X1).asarray()


def der_sigmoid(X) :
    X1 = cm.CUDAMatrix(sigmoid(X))
    R = cm.CUDAMatrix(sigmoid(X))
    R1 = cm.CUDAMatrix(np.ones(X.shape))

    R1.subtract(R)
    X1.mult(R1)

    return X1.asarray()


def der_relu(X) :
    X1 = cm.CUDAMatrix(X)

    X1.greater_than(0)

    return X1.asarray()


def der_tanh(X) :
    X1 = cm.CUDAMatrix(X)
    R1 = cm.CUDAMatrix(np.ones(X.shape))
    Ex = cm.empty(X.shape)
    EX = cm.empty(X.shape)

    Ex = cm.exp(X)
    EX = cm.exp(cm.empty(X.shape).subtract(X1))

    R1 = Ex.add(EX).mult(0.5)

    return R1.asarray()