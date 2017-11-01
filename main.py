import core

X, Y = core.load_train_set()

print(core.nn_model(X, Y, layers_dims=[3072, 300, 100, 50, 20, 10, 1], print_cost=True, learning_rate=0.000000002))