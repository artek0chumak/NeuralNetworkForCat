import tensorflow as tf
import numpy as np
import nn_core
import pickle

def load_set(file):
    X_ = list()
    Y_ = list()
    with open(file, 'rb') as img:
        data = pickle.load(img, encoding='bytes')

    X = np.array(data[b'data']).T
    Y = tf.one_hot(np.array(data[b'fine_labels']), max(data[b'fine_labels']) + 1, dtype=tf.float32)
    Y = tf.transpose(Y)

    sess = tf.Session()
    Y = sess.run(Y)
    sess.close()

    return X, Y

def model(X_train, Y_train, X_test, Y_test, layers_dim, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X = tf.placeholder(shape=(n_x, None), dtype=tf.float32)
    Y = tf.placeholder(shape=(n_y, None), dtype=tf.float32)

    parameters = nn_core.initialization(layers_dim)
    ZL = nn_core.forward_propagation(X, parameters, ["relu", "relu", "relu", "relu"])

    cost = nn_core.compute_cost(ZL, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.device("/cpu:0"):
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                epoch_cost = 0.
                num_minibatches = int(m / minibatch_size)
                minibatches = nn_core.random_mini_batches(X_train, Y_train, minibatch_size)

                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    epoch_cost += minibatch_cost / num_minibatches

                if print_cost == True and epoch % 100 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)

        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

X_train, Y_train = load_set("cifar-100-python/train")
X_test, Y_test = load_set("cifar-100-python/test")

parameters = model(X_train, Y_train, X_test, Y_test, [X_train.shape[0], 200, 100, 100, Y_train.shape[0]])
