import pickle
import numpy as np

def load_set_class(file):
    X_ = list()
    Y_ = list()
    for i in file:
        with open(i, 'rb') as img:
            data = pickle.load(img, encoding='bytes')
        for j in range(len(data[b'data'])):
            m = np.max(data[b'data'])
            X_.append(data[b'data'][j] / m)
            a = [0]*(max(data[b'labels']) - min(data[b'labels']) + 1)
            a[data[b'labels'][j]] = 1
            Y_.append(a)

    return np.array(X_).T, np.array(Y_).reshape((len(a), len(file) * len(data[b"data"])))


def load_set_linear(file):
    X_ = list()
    Y_ = list()
    for i in file:
        with open(i, 'rb') as img:
            data = pickle.load(img, encoding='bytes')
        for j in range(len(data[b'data'])):
            X_.append(data[b'data'][j])
            Y_.append(data[b'labels'][j])

    return np.array(X_).T, np.array(Y_).reshape((1, len(file) * len(data[b"data"])))


def load_train_set():
    X_ = list()
    Y_ = list()
    for i in range(5):
        with open('cifar-10-batches-py/data_batch_{0}'.format(i + 1), 'rb') as img:
            data = pickle.load(img, encoding='bytes')
        for j in range(10000):
            X_.append(data[b'data'][j] / 256)
            a = [0]*10
            a[data[b'labels'][j]] = 1
            Y_.append(a)

    return np.array(X_).T, np.array(Y_).reshape((10, 50000))