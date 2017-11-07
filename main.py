import nn_load
import nn_main

file = list()

for i in range(5):
    file.append("cifar-10-batches-py/data_batch_" + str(i+1))

X, Y = nn_load.load_train_set()
dims = [3072, 200, 80, 40, 20, 10]
nn_main.nn_model(X, Y, dims, print_cost=True, learning_rate=0.1, num_iteration=1000)