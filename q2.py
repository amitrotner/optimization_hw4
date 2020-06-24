import mcholmz
import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt
import time


def plot_graph(iter, y_values):
    x = range(iter)
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].plot(x, y_values[0])
    axs[0, 0].set_title('Gradient of the Augmented\nLagrangian aggregate')
    axs[0, 1].plot(x,  y_values[1], 'bo')
    axs[0, 1].set_title('Maximal constraint\nviolation')
    axs[0, 2].plot(x, y_values[2])
    axs[0, 2].set_title('Residual in the\nobjective function')
    axs[1, 0].plot(x, y_values[3])
    axs[1, 0].set_title('Distance to the\n optimal point')
    axs[1, 1].plot(x, y_values[4])
    axs[1, 1].set_title('Distance to the\noptimal multipliers')
    fig.delaxes(axs[1, 2])

    for i, ax in enumerate(axs.flat):
        if i != 1:
            ax.set(yscale='log')
    plt.show()


def newton(f, lambda_k, der_f, hes_f, x0, pk, constrains, alpha0, sigma, beta, epsilon):
    L, D, e = mcholmz.modifiedChol(hes_f(x0, constrains, pk, lambda_k))
    grad = der_f(x0, constrains, pk, lambda_k)
    y = scipy.linalg.solve_triangular(-L, grad, lower=True)
    d = scipy.linalg.solve_triangular(np.matmul(np.diagflat(D), L.T), y, lower=False)
    x = x0
    while np.linalg.norm(der_f(x, constrains, pk, lambda_k)) >= epsilon:
        alpha = alpha0
        F_armijo = (f(x + alpha * d, constrains, pk, lambda_k) - f(x, constrains, pk, lambda_k))
        F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x, constrains, pk, lambda_k).T, d))
        while float(F_armijo) > float(F_armijo_sigma):
            alpha = beta * alpha
            F_armijo = (f(x + alpha * d, constrains, pk, lambda_k) - f(x, constrains, pk, lambda_k))
            F_armijo_sigma = (sigma * alpha * np.matmul(der_f(x, constrains, pk, lambda_k).T, d))
        x = x + alpha * d
        L, D, e = mcholmz.modifiedChol(hes_f(x, constrains, pk, lambda_k))
        grad = der_f(x, constrains, pk, lambda_k)
        y = scipy.linalg.solve_triangular(-L, grad, lower=True)
        d = scipy.linalg.solve_triangular(np.matmul(np.diagflat(D), L.T), y, lower=False)
    return x


def f(x):
    Q = np.array([[4, 0], [0, 2]])
    d = np.array([-20, -2])
    return 0.5 * np.matmul(np.matmul(x.T, Q), x) + np.matmul(d.T, x) + 51


def der_f(x):
    Q = np.array([[4, 0], [0, 2]])
    d = np.array([-20, -2])
    return np.matmul(Q, x) + d


def hess_f(x):
    Q = np.array([[4, 0], [0, 2]])
    return Q


def constrain1(x):
    A = np.array([[0.5, 1], [1, -1], [-1, -1]])
    b = np.array([1, 0, 0])
    return np.matmul(A, x) - b


def grad_constrain1():
    A = np.array([[0.5, 1], [1, -1], [-1, -1]])
    return A.T


def phi_p(x, p):
    x = p * x
    x = (x ** 2) / 2 + x if x >= -0.5 else -0.25 * np.log(-2 * x) - (3 / 8)
    return 1 / p * x


def der_phi_p(x, u):
    x = x * u
    result = x + 1 if x >= -0.5 else -1 / (4 * x)
    return result


def F_p(x, constrains: list, p, lambda_k):
    return f(x) + np.sum([phi_p(constrains[0](x)[i], p * lambda_k[i]) for i in range(len(lambda_k))])


def der_F_p(x, constrains: list, p, lambda_k):
    result = der_f(x)
    for i in range(len(lambda_k)):
        result += der_phi_p(constrains[0](x)[i], p * lambda_k[i]) * grad_constrain1()[:, i]
    return result


def hess_F_p(x, constrains: list, p, lambda_k):
    return hess_f(x)


def augmented_lagrangian(constrains: list):
    pk = 1
    p_max = 800
    alpha = 2
    alpha0 = 1
    sigma = 0.25
    beta = 0.5
    epsilon = 1e-5
    x_k = np.ones(2)
    lambda_k = np.ones(3)
    iter = 0
    l2_norm = []
    maximal_violation = []
    residual_objective = []
    optimal_point_dist = []
    optimal_multipliers_dist = []
    while pk < p_max:
        x_k = newton(F_p, lambda_k, der_F_p, hess_F_p, x_k, pk, constrains, alpha0, sigma, beta, epsilon)
        for i in range(len(lambda_k)):
            lambda_k[i] = der_phi_p(constrains[0](x_k)[i], pk * lambda_k[i])
        pk = min(p_max, alpha * pk)
        iter += 1
        l2_norm += [np.linalg.norm(der_F_p(x_k, constrains, pk, lambda_k))]
        maximal_violation += [np.argmax(constrain1(x_k)) + 1]
        residual_objective += [abs(f(np.array([2 / 3, 2 / 3])) - f(x_k))]
        optimal_point_dist += [np.linalg.norm(x_k - np.array([2 / 3, 2 / 3]))]
        optimal_multipliers_dist += [np.linalg.norm(lambda_k - np.array([12, 34 / 3, 0]))]
    plot_graph(iter, [l2_norm,maximal_violation,residual_objective,optimal_point_dist,optimal_multipliers_dist])
    return x_k, lambda_k


if __name__ == '__main__':
    constrains = [constrain1]
    x_star, lambda_star = augmented_lagrangian(constrains)
    print(x_star, lambda_star)
