import core

X, Y = core.load_train_set()

print(core.nn_model(X, Y, layers_dims=[3072, 20, 5, 1], print_cost=True, learning_rate=1.2))