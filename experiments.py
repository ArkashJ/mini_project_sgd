import random
from typing import Optional

import autograd.numpy as np
import scipy as sp

from sgd_robust_regression import *

# define constants
NU: int = 5
ETA: float = 0.2
ETA_0: float = 0.2
ALPHA: float = 0.51
B: int = 10
N: int = 10000
D: int = 10


def generate_params() -> tuple:
    generate_seed = random.seed(100)
    true_beta, Y, Z = generate_data(N, D, generate_seed)
    return true_beta, Y, Z


def get_sgd_and_sgd_grad(
    Y: np.ndarray,
    Z: np.ndarray,
    NU: int,
) -> tuple:
    sgd_loss, grad_sgd = make_sgd_robust_loss(Y, Z, NU)
    return sgd_loss, grad_sgd


def estimate_x_star(
    sgd_loss: np.ndarray,
    init_param: np.ndarray,
) -> Optional[np.ndarray]:
    try:
        res_minimize = sp.optimize.minimize(
            sgd_loss,
            init_param,
            args=(np.arange(N),),
        )
        return res_minimize.x
    except ValueError as e:
        print("Error in estimate_x_star - ", e)
        status_code = 500
        return e, status_code


def get_params(epochs, grad_loss, init_param) -> np.ndarray:
    params = run_sgd(
        grad_loss,
        epochs,
        init_param=init_param,
        init_stepsize=ETA,
        stepsize_decayrate=0,
        batchsize=B,
        n=N,
    )
    # params is x_k itself.
    return params


# Get the average of the last k/2 iterations
def get_iterate_average(params: np.ndarray) -> np.ndarray:
    iterate_average = np.array(
        [np.mean(params[i // 2 : i + 1], axis=0) for i in range(params.shape[0])]
    )
    return iterate_average


# run multiple epochs and plot the results
def run_multiple_epochs():
    epochs_list = [5, 10, 15, 20, 25, 50, 100]
    init_param = np.zeros(D + 1)
    true_beta, Y, Z = generate_params()
    sgd_loss, grad_sgd = get_sgd_and_sgd_grad(Y, Z, NU)
    x_star = estimate_x_star(sgd_loss, init_param)

    for epochs in epochs_list:
        params = get_params(epochs, grad_sgd, init_param)
        x_k = params
        iterate_average = get_iterate_average(params)
        plot_iterates_and_squared_errors(
            x_k, true_beta, x_star, 0, epochs, N, B, "x_k_{}".format(epochs), True
        )

        plot_iterates_and_squared_errors(
            iterate_average,
            true_beta,
            x_star,
            0,
            epochs,
            N,
            B,
            "iterate_average_{}".format(epochs),
            True,
        )


def run_different_initialization():
    multiplication_factor_for_init_param_vec = [
        -10,
        -5,
        -2,
        -0.1,
        0.1,
        0.5,
        1,
        2,
        5,
        10,
        25,
        50,
    ]

    results_dict = {}
    epochs = 20
    for factor in multiplication_factor_for_init_param_vec:
        init_param = np.ones(D + 1) * factor
        true_beta, Y, Z = generate_params()
        sgd_loss, grad_sgd = get_sgd_and_sgd_grad(Y, Z, NU)
        x_star = estimate_x_star(sgd_loss, init_param)

        params = get_params(epochs, grad_sgd, init_param)
        x_k = params
        iterate_average = get_iterate_average(params)
        print("printing for factor - ", factor)
        distance_between_init_param_and_optimal = (
            np.linalg.norm(init_param - x_star[np.newaxis, :], axis=1) ** 2
        )
        results_dict[factor] = distance_between_init_param_and_optimal
        plot_iterates_and_squared_errors(
            x_k,
            true_beta,
            x_star,
            0,
            epochs,
            N,
            B,
            "x_k_{}_{}".format(epochs, factor),
            True,
        )

        distance_between_init_param_and_optimal = (
            np.linalg.norm(init_param - x_star[np.newaxis, :], axis=1) ** 2
        )
        results_dict[factor] = distance_between_init_param_and_optimal
        plot_iterates_and_squared_errors(
            iterate_average,
            true_beta,
            x_star,
            0,
            epochs,
            N,
            B,
            "iterate_average_{}_{}".format(epochs, factor),
            True,
        )


def main():
    # run_multiple_epochs()
    run_different_initialization()


main()
    
