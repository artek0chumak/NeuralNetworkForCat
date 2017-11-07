import nn_under
import cudamat as cm

def nn_model(X, Y, layers_dims, num_iteration=10000, print_cost=False, learning_rate=0.002):
    cm.cuda_set_device(0)
    cm.init()

    parameters = nn_under.initialize_parameters(layers_dims)

    for i in range(num_iteration):
        AL, caches = nn_under.forward_propagation(X, parameters)
        cost = nn_under.compute_cost(AL, Y, parameters)
        grads = nn_under.backward_propagation(AL, Y, caches)
        parameters = nn_under.update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {0}: {1}".format(i, cost))

    return parameters

    cm.shutdown()