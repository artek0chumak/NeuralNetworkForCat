import nn_under
import numpy as np
import cudamat as cm

def grad_check(X, Y, parameters, grads, epsilon=0.000000001):
    J_plus = np.zeros((len(parameters), 1))
    J_minus = np.zeros((len(parameters), 1))
    grad_ap = np.zeros((len(parameters), 1))
    grad_np = np.zeros((len(parameters), 1))
    for i in range(len(grads)):
        grad_np[i] = grads[i]

    for l in range(len(parameters) // 2):
        theta_plus = parameters
        theta_plus["W" + str(l+1)] = theta_plus["W" + str(l+1)] + epsilon
        J_plus[l], _ = nn_under.compute_cost(nn_under.forward_propagation(X, theta_plus)[0], Y, theta_plus)

        theta_minus = parameters
        theta_minus["W" + str(l + 1)] = theta_plus["W" + str(l + 1)] - epsilon
        J_minus[l], _ = nn_under.compute_cost(nn_under.forward_propagation(X, theta_minus)[0], Y, theta_minus)

        grad_ap[l] = (J_plus[l] - J_minus[l]) / 2 / epsilon

        theta_plus = parameters
        theta_plus["b" + str(l + 1)] = theta_plus["b" + str(l + 1)] + epsilon
        J_plus[l * 2], _ = nn_under.compute_cost(nn_under.forward_propagation(X, theta_plus)[0], Y, theta_plus)

        theta_minus = parameters
        theta_minus["b" + str(l + 1)] = theta_plus["b" + str(l + 1)] - epsilon
        J_minus[l * 2], _ = nn_under.compute_cost(nn_under.forward_propagation(X, theta_minus)[0], Y, theta_minus)

        grad_ap[l * 2] = (J_plus[l * 2] - J_minus[l * 2]) / 2 / epsilon

    numerator = np.linalg.norm(grad_np - grad_ap)
    denominator = np.linalg.norm(grad) + np.linalg.norm(grad_ap)
    difference = numerator / denominator

    if difference > 2e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference