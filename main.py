import nn_load
import nn_main
import numpy as np

# file = list()
#
# for i in range(5):
#     file.append("cifar-10-batches-py/data_batch_" + str(i+1))

file = ["cifar-100-python/train"]

X, Y = nn_load.load_set_class(file)
X = X / np.max(X)
dims = [X.shape[0], 200, 80, 40, 20, Y.shape[0]]
nn_main.nn_model(X, Y, dims, print_cost=True, learning_rate=0.1, num_iteration=1000)