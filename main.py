import core
import cuda_core


X, Y = cuda_core.load_train_set()

print(cuda_core.nn_model(X, Y, layers_dims=[3072, 100, 100, 50, 20, 10, 1], print_cost=True, learning_rate=0.0000000005))
