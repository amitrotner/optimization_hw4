import mcholmz
import numpy as np
import scipy
from scipy import io
import matplotlib.pyplot as plt
import time


def phi(x):
    return np.tanh(x)


def phi_deriv(x):
    return 1 - (np.tanh(x) ** 2)


def f(x1, x2):
    return x1 * (np.exp(-(x1 ** 2 + x2 ** 2)))


def F(x, W1, W2, W3, b1, b2, b3):
    u1 = np.matmul(W1.T, x) + b1
    u2 = np.matmul(W2.T, phi(u1)) + b2
    res = np.matmul(W3.T, phi(u2)) + b3
    return res


def psi(r):
    return r ** 2


def psi_deriv(r):
    return 2*r


def error(*kwargs):
    args = kwargs[0]
    return psi(F(args[0], args[1], args[2], args[3], args[4], args[5], args[6]) - f(args[0][0], args[0][1]))


def error_grad_W1(x, W1, W2, W3, b1, b2, b3):
    phi1 = phi_deriv(np.matmul(W1.T, x) + b1)
    phi1_tag = np.diagflat(phi1)
    phi2 = phi_deriv(np.matmul(W2.T, phi(np.matmul(W1.T, x) + b1)) + b2)
    phi2_tag = np.diagflat(phi2)
    fx = F(x, W1, W2, W3, b1, b2, b3)
    y = f(x[0], x[1])
    r = fx - y
    elem1 = np.matmul(x*psi_deriv(r), W3.T)
    elem2 = np.matmul(elem1, phi2_tag)
    elem3 = np.matmul(elem2, W2.T)
    res = np.matmul(elem3, phi1_tag)
    return res


def error_grad_b1(x, W1, W2, W3, b1, b2, b3):
    phi1 = phi_deriv(np.matmul(W1.T, x) + b1)
    phi1_tag = np.diagflat(phi1)
    phi2 = phi_deriv(np.matmul(W2.T, phi(np.matmul(W1.T, x) + b1)) + b2)
    phi2_tag = np.diagflat(phi2)
    r = F(x, W1, W2, W3, b1, b2, b3) - f(x[0], x[1])
    res = np.matmul(np.matmul(np.matmul(np.matmul(phi1_tag, W2), phi2_tag), W3), psi_deriv(r))
    return res


def error_grad_W2(x, W1, W2, W3, b1, b2, b3):
    phi1 = phi(np.matmul(W1.T, x) + b1)
    phi2 = phi_deriv(np.matmul(W2.T, phi(np.matmul(W1.T, x) + b1)) + b2)
    phi2_tag = np.diagflat(phi2)
    r = F(x, W1, W2, W3, b1, b2, b3) - f(x[0], x[1])
    res = np.matmul(np.matmul(np.matmul(phi1, psi_deriv(r)), W3.T), phi2_tag)
    return res


def error_grad_b2(x, W1, W2, W3, b1, b2, b3):
    phi2 = phi_deriv(np.matmul(W2.T, phi(np.matmul(W1.T, x) + b1)) + b2)
    phi2_tag = np.diagflat(phi2)
    r = F(x, W1, W2, W3, b1, b2, b3) - f(x[0], x[1])
    res = np.matmul(np.matmul(phi2_tag, W3), psi_deriv(r))
    return res


def error_grad_W3(x, W1, W2, W3, b1, b2, b3):
    phi2 = phi(np.matmul(W2.T, phi(np.matmul(W1.T, x) + b1)) + b2)
    r = F(x, W1, W2, W3, b1, b2, b3) - f(x[0], x[1])
    res = np.matmul(phi2, psi_deriv(r))
    return res


def error_grad_b3(x, W1, W2, W3, b1, b2, b3):
    r = F(x, W1, W2, W3, b1, b2, b3) - f(x[0], x[1])
    res = psi_deriv(r)
    return res


def error_grad(i, kwargs):
    if i == 1:
        return error_grad_W1(*kwargs)
    elif i == 2:
        return error_grad_W2(*kwargs)
    elif i == 3:
        return error_grad_W3(*kwargs)
    elif i == 4:
        return error_grad_b1(*kwargs)
    elif i == 5:
        return error_grad_b2(*kwargs)
    elif i == 6:
        return error_grad_b3(*kwargs)


def create_numeric_grad(ind, *kwargs):
    eps = ((2 * 10e-16) ** (1/3))
    elem_plus_eps = list(*kwargs)
    elem_minus_eps = list(*kwargs)
    elem_to_grad = elem_plus_eps[ind]
    flattened_elem_to_grad = elem_to_grad.flatten()
    grad = np.zeros_like(flattened_elem_to_grad)
    for i, val in enumerate(flattened_elem_to_grad):
        new_elem_plus = np.copy(flattened_elem_to_grad)
        new_elem_plus[i] = flattened_elem_to_grad[i] + eps
        new_elem_minus = np.copy(flattened_elem_to_grad)
        new_elem_minus[i] = flattened_elem_to_grad[i] - eps
        elem_plus_eps[ind] = np.reshape(new_elem_plus, elem_to_grad.shape)
        elem_minus_eps[ind] = np.reshape(new_elem_minus, elem_to_grad.shape)
        grad[i] = (error(elem_plus_eps) - error(elem_minus_eps)) / (2*eps)
    res = np.reshape(grad, elem_to_grad.shape)
    return res


def calc_grads(args):
    for i, val in enumerate(args):
        analytic_grad = error_grad(i, args)


def plot_grad_diff_graphs(x, W1, W2, W3, b1, b2, b3):
    kwargs = [x, W1, W2, W3, b1, b2, b3]
    diffs = []
    for i, val in enumerate(kwargs):
        if i == 0:
            continue
        numeric_grad = create_numeric_grad(i, kwargs)
        analytic_grad = error_grad(i, kwargs)
        grad_diff = analytic_grad - numeric_grad

        plt.plot(grad_diff.flatten(), 'o')
        plt.xlabel('Coordinate')
        plt.ylabel('Value')
        plt.show()

        plt.imshow(grad_diff)
        plt.colorbar()
        plt.show()
        diffs.append(grad_diff)


def plot_objective_func():
    x = np.arange(-2, 2, 0.2)
    y = np.arange(-2, 2, 0.2)
    xx, yy = np.meshgrid(x, y)
    z = f(xx, yy)
    h = plt.contourf(x, y, z)
    plt.show()
    print('printed')


def generate_set(Ntrain):
    x = 4 * np.random.rand(Ntrain, 2) - 2
    return x


def psi_train(set, W1, W2, W3, b1, b2, b3, calc_grads=False):
    err_sum = 0
    W1_sum = np.zeros_like(W1)
    W2_sum = np.zeros_like(W2)
    W3_sum = np.zeros_like(W3)
    b1_sum = np.zeros_like(b1)
    b2_sum = np.zeros_like(b2)
    b3_sum = np.zeros_like(b3)
    set_size = set.shape[0]
    for i in range(set_size):
        err_sum += error([np.array([set[i]]).T, W1, W2, W3, b1, b2, b3])
        if calc_grads:
            W1_sum += error_grad_W1(np.array([set[i]]).T, W1, W2, W3, b1, b2, b3)
            W2_sum += error_grad_W2(np.array([set[i]]).T, W1, W2, W3, b1, b2, b3)
            W3_sum += error_grad_W3(np.array([set[i]]).T, W1, W2, W3, b1, b2, b3)
            b1_sum += error_grad_b1(np.array([set[i]]).T, W1, W2, W3, b1, b2, b3)
            b2_sum += error_grad_b2(np.array([set[i]]).T, W1, W2, W3, b1, b2, b3)
            b3_sum += error_grad_b3(np.array([set[i]]).T, W1, W2, W3, b1, b2, b3)
    grads = np.array([W1_sum, W2_sum, W3_sum, b1_sum, b2_sum, b3_sum])
    return err_sum/set_size, grads/set_size


def plot_graph(figure_title, iter, convergence_curve, save_name):
    plt.plot(range(iter), convergence_curve)
    plt.title(figure_title)
    plt.ylabel(r"$f(x_{k})-f^*$")
    plt.xlabel("Iteration Number")
    plt.yscale('log')
    plt.savefig('graphs/' + save_name + '.svg', format='svg')
    plt.show()


def make_flat(args):
    flatten = np.concatenate((args[0], args[1], args[2], args[3], args[4], args[5]), axis=None)
    return flatten


def reshape_grads(to_reshape, shapes):
    ret_grads = []
    start_ind = 0
    for arg in shapes:
        g = np.array(to_reshape[start_ind:(start_ind + arg.size)])
        g = np.reshape(g, arg.shape)
        ret_grads.append(g)
        start_ind += arg.size
    return ret_grads


def BFGS(target_func, set, args, alpha0, sigma, beta, epsilon):
    start = time.time()
    c2 = 0.9
    err_val, grad = target_func(set, *args, calc_grads=True)
    iter = 0
    flattened_grads = make_flat(grad)
    B = np.identity(flattened_grads.size)
    convergence_curve = []
    while np.linalg.norm(flattened_grads) >= epsilon:
        d = np.matmul(-B, flattened_grads)
        convergence_curve.append(err_val)
        alpha = alpha0
        flattened_args = make_flat(args)
        flattened_args_inc = flattened_args + alpha*d
        args_inc = reshape_grads(flattened_args_inc, args)
        F_armijo = (target_func(set, *args_inc)[0] - target_func(set, *args)[0])
        F_armijo_sigma = (sigma * alpha * np.matmul(flattened_grads.T, d))
        while F_armijo > F_armijo_sigma:
            alpha = beta * alpha
            flattened_args_inc = flattened_args + alpha * d
            args_inc = reshape_grads(flattened_args_inc, args)
            F_armijo = (target_func(set, *args_inc)[0] - target_func(set, *args)[0])
            F_armijo_sigma = (sigma * alpha * np.matmul(flattened_grads.T, d))
        prev_args = np.copy(args)
        prev_grad = np.copy(grad)
        flattened_args = flattened_args + alpha * d
        args = reshape_grads(flattened_args, args)
        err_val, grad = target_func(set, *args, calc_grads=True)
        flattened_grads = make_flat(grad)
        p = flattened_args - make_flat(prev_args)
        # if not np.array_equal(p, alpha*d):
        #     print(iter)
        q = flattened_grads - make_flat(prev_grad)
        s = np.matmul(B, q)
        tau = np.matmul(s.T, q)
        mu = np.matmul(p.T, q)
        v = (1 / mu) * p - (1 / tau) * s
        if abs(mu) < 10e-20:
            B = np.identity(flattened_args.size)
            print("reset B")
        # elif abs(mu) < (10e-6) * np.matmul(p.T, np.matmul(B, p)):
        #     print("continue")
        if np.matmul(flattened_grads.T, d) > c2 * np.matmul(make_flat(prev_grad).flatten().T, d) and mu != 0 and tau != 0:
            B = B + ((1 / mu) * np.matmul(p, p.T)) - ((1 / tau) * np.matmul(s, s.T)) + tau * np.matmul(v, v.T)
            """H = np.linalg.inv(rosen_hess(x))
            print(B - H)"""
        iter += 1
        print(err_val)
        if not iter % 50:
            print("grad norm" + str(np.linalg.norm(flattened_grads)))
    end = time.time()
    # plot_graph(figure_title + "\nTotal running time: " + str(("{:.5f}".format(end - start))) + " sec", iter,
    #            convergence_curve, save_name)
    return err_val, args


def run_and_plot_test_set(set, args):
    res_vec = []
    set_size = set.shape[0]
    for i in range(set_size):
        res_vec.append(F(np.array([set[i]]).T, args[0], args[1], args[2], args[3], args[4], args[5]))

    X = set.T[0]
    Y = set.T[1]
    ax = plt.axes(projection='3d')
    network_reconstruction = np.array(res_vec).flatten()
    ax.plot_trisurf(X, Y, network_reconstruction, linewidth=0, antialiased=False)
    ax.set_title('surface')
    plt.show()



if __name__ == "__main__":
    training_set = generate_set(500)
    test_set = generate_set(200)
    W1 = np.array(np.random.rand(2, 4) / np.sqrt(2))
    W2 = np.array(np.random.rand(4, 3) / 2)
    W3 = np.array(np.random.rand(3, 1) / np.sqrt(3))
    b1 = np.zeros((4, 1))
    b2 = np.zeros((3, 1))
    b3 = np.zeros((1, 1))
    args = [W1, W2, W3, b1, b2, b3]
    alpha0 = 1
    sigma = 0.25
    beta = 0.5
    epsilon = 1e-5
    err_val, new_args = BFGS(psi_train, training_set, args, alpha0, sigma, beta, epsilon)
    run_and_plot_test_set(test_set, new_args)
    # plot_grad_diff_graphs(x, W1, W2, W3, b1, b2, b3)
    # plot_objective_func()

    print('done')

