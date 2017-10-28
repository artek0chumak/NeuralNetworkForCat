import core

X, Y = core.load_data_set()

print(core.nn_model(X, Y, 20, 5, print_cost=True))