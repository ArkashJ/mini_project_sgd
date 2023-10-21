from autograd import grad
import autograd.numpy as np
from sgd_robust_regression import *
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from typing import Optional
import random
import math

NU: int = 5
ETA: float = 0.2
ETA_0: float = 5
ALPHA: float = 0.51
B: int = 10
N: int = 10000
D: int = 10


def print_constants() -> None:
    print(
        f"Batch size: {B}\n"
        f"Number of samples: {N}\n"
        f"Number of features: {D}\n"
        f"Degrees of freedom: {NU}\n"
        f"Initial step size: {ETA_0}\n"
        f"Decay rate: {ALPHA}\n"
        f"step size: {ETA}\n"
    )


class run_experiments:
    def __init__(self, N, D, NU, ETA, ETA_0, ALPHA, B):
        self.N = N
        self.D = D
        self.NU = NU
        self.ETA = ETA
        self.ETA_0 = ETA_0
        self.ALPHA = ALPHA
        self.B = B

    def init_param_test(self) -> None:
        true_beta, Y, Z = generate_data(self.N, self.D, 0)
        sgd_loss, grad_sgd_loss = make_sgd_robust_loss(Y, Z, self.NU)
        init_param = np.zeros(self.D + 1)
        paramiters = run_SGD(grad_sgd_loss, 10, init_param, self.ETA, 0, self.B, self.N)
        print(
            "Final paramiters- \n",
            paramiters,
            "Size of paramiters- \n",
            paramiters.shape,
        )
    
    def generate_param_and_sgd(self) -> tuple:
        generate_seed = random.randint(0, 100)
        true_beta, Y, Z = generate_data(self.N, self.D, generate_seed)
        sgd_loss, grad_sgd_loss = make_sgd_robust_loss(Y, Z, self.NU)
        init_param = np.zeros(self.D + 1)
        return sgd_loss, grad_sgd_loss, init_param

    def estimate_x_tilda_k(
        self,
        grad_loss,
        batchsize,
        n,
        stepsize,
        epochs: int = 5,
        x=np.zeros(D + 1),
        alpha=0,
        avg_range: float = 0.5,
    ) -> Optional[np.ndarray]:
        params = run_sgd(grad_loss, epochs, x, stepsize, alpha, batchsize, n)

        x_iterate = params[-1]
        print("x_iterate- \n", x_iterate)

        k = epochs * n // batchsize
        lower_range, upper_range = math.floor(k * avg_range), k + 1
        for i in range(lower_range, upper_range):
            x_iterate += params[i]
        x_iterate_average = x_iterate / (upper_range - lower_range)
        print("x_iterate_average- \n", x_iterate_average)

        return x_iterate, x_iterate_average

    def find_norm(
        self,
        x_star: np.ndarray,
        x_iterate: np.ndarray,
        x_iterate_average: np.ndarray,
    ) -> float:
        return np.linalg.norm(x_star - x_iterate), np.linalg.norm(
            x_star - x_iterate_average
        )

    # x_start = argmin (1/N)* sum(robust_loss(psi, beta, nu, Y, Z)) + (1/2N)*sum(beta**2)
    def estimate_x_star(self) -> Optional[np.ndarray]:
        try:
            # book page 7, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
            sgd_loss, grad_sgd_loss, init_param = self.generate_param_and_sgd()

            res_minimize = sp.optimize.minimize(
                sgd_loss,
                init_param,
                args=(np.arange(self.N),),
            )
            print("res_minimize- \n", res_minimize)
            print(f"res_minimize.x- \n {res_minimize.x}")

            return res_minimize.x, grad_sgd_loss
        except ValueError as e:
            print("Error in estimate_x_star")
            print(e)
    
    def test_norms(self):

        x_star, grad_sgd_loss = self.estimate_x_star()
        # call estimate_x_tilda_k
        x_iterate, x_iterate_average = self.estimate_x_tilda_k(
            grad_sgd_loss,
            self.B,
            self.N,
            self.ETA,
        )
        norm_x_star, norm_x_tilda_k = self.find_norm(
            x_star, x_iterate, x_iterate_average
        )
        print(f"norm_x_star- \n {norm_x_star}")
        print(f"norm_x_tilda_k- \n {norm_x_tilda_k}")


def main():
    experiments = run_experiments(N, D, NU, ETA, ETA_0, ALPHA, B)
    # experiments.init_param_test()
    experiments.test_norms()


main()
