import autograd.numpy as np
from sgd_robust_regression import *
import scipy as sp
from typing import Optional
import random
import math
from tqdm import tqdm

# define constants
NU: int = 5
ETA: float = 0.2
ETA_0: float = 5
ALPHA: float = 0.51
B: int = 10
N: int = 10000
D: int = 10


# class to run experiments
class run_experiments:
    def __init__(self, N, D, NU, ETA, ETA_0, ALPHA, B):
        self.N = N
        self.D = D
        self.NU = NU
        self.ETA = ETA
        self.ETA_0 = ETA_0
        self.ALPHA = ALPHA
        self.B = B

    def print_constants(self) -> None:
        print(
            f"Batch size: {self.B}\n"
            f"Number of samples: {self.N}\n"
            f"Number of features: {self.D}\n"
            f"Degrees of freedom: {self.NU}\n"
            f"Initial step size: {self.ETA_0}\n"
            f"Decay rate: {self.ALPHA}\n"
            f"step size: {self.ETA}\n"
        )

    # Get the sgd_loss, grad_sgd_loss for generated data
    def generate_param_and_sgd(self) -> tuple:
        generate_seed = random.randint(0, 100)
        true_beta, Y, Z = generate_data(self.N, self.D, generate_seed)
        sgd_loss, grad_sgd_loss = make_sgd_robust_loss(Y, Z, self.NU)
        init_param = np.zeros(self.D + 1)
        return sgd_loss, grad_sgd_loss, init_param

    # Helper function to find the norm
    def find_norm(
        self,
        x_star: np.ndarray,
        iterate_params: np.ndarray,
    ) -> float:
        x_iterate = np.linalg.norm(x_star - iterate_params)
        return x_iterate

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

            print(f"res_minimize.x- \n {res_minimize.x}")

            return res_minimize.x, grad_sgd_loss
        except ValueError as e:
            print("Error in estimate_x_star")
            print(e)

    # Find the norm^2 for x_iterate and x_iterate_average
    def estimate_x_tilda_k(
        self,
        grad_loss: np.ndarray,
        batchsize: int,
        n: int,
        stepsize: int,
        epochs: int,
        x_star: np.ndarray,
        x=np.zeros(D + 1),
        alpha=0,
        avg_range: float = 0.5,
    ) -> np.ndarray:
        params = run_sgd(grad_loss, epochs, x, stepsize, alpha, batchsize, n)

        # find the range of the last 50% of the iterations
        length = len(params)
        x_iterate_average = (
            self.find_norm(x_star, np.mean(params[length // 2 :]))
        ) ** 2
        x_iterate = self.find_norm(x_star, params[length - 1]) ** 2
        print(f"The Norm for x_iterate is - \n {x_iterate}")
        print(f"The Norm for iterate average is - \n {x_iterate_average}")
        print("\n")
        return x_iterate, x_iterate_average

    # Run multiple experiments to find the best number of epochs
    def find_best_num_epochs(self) -> None:
        epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200, 500]
        x_star, grad_sgd_loss = self.estimate_x_star()
        for epoch in tqdm(epochs):
            print(f"epoch- \n {epoch}")
            x_iterate, x_iterate_average = self.estimate_x_tilda_k(
                grad_sgd_loss,
                self.B,
                self.N,
                self.ETA,
                epoch,
                x_star,
            )

    # Given a list of initializations, find the norm^2 for each of them
    def test_initialization(self, init_param_vec: np.ndarray) -> None:
        norms_vec = np.zeros(len(init_param_vec))
        x_star, grad_sgd_loss = self.estimate_x_star()
        max_change, min_change = 0, 0
        max_norm, min_norm = 0, 0

        len_arr = len(init_param_vec)
        for i in tqdm(range(len_arr)):
            norms_vec[i] = self.find_norm(x_star, init_param_vec[i]) ** 2
            if i == 0:
                min_norm = norms_vec[i]
            if i > 0:
                if norms_vec[i] - norms_vec[i - 1] > max_change:
                    max_change = norms_vec[i] - norms_vec[i - 1]
                if norms_vec[i] - norms_vec[i - 1] < min_change:
                    min_change = norms_vec[i] - norms_vec[i - 1]
            if norms_vec[i] > max_norm:
                max_norm = norms_vec[i]
            if norms_vec[i] < min_norm:
                min_norm = norms_vec[i]

            # print(
            #     f"Iteration {i}\t, norm is {norms_vec[i]} \t, \ninit_param is {init_param_vec[i]}\n"
            # )
        average_change = np.mean(norms_vec)
        # find quintiles
        print(
            f"\nQuintiles are as follows: \n {np.quantile(norms_vec, [0.2, 0.4, 0.6, 0.8])}"
        )

        print(
            f"\nStatistics are as follows: \n max_change: {max_change} \n min_change: {min_change} \n average_norm: {average_change} \n min_norm: {min_norm} \n max_norm: {max_norm}\n"
        )

    # Run multiple experiments to find the best initialization
    def find_best_initilization_param(self) -> None:
        init_param = np.zeros(self.D + 1)
        init_param_multivariate_normal = np.array(
            [
                np.random.multivariate_normal(init_param, np.identity(self.D + 1))
                for i in range(1000)
            ]
        )
        init_param_uniform = np.array(
            [np.random.uniform(-1, 1, self.D + 1) for i in range(1000)]
        )

        print("---------------- \n Testing out values with multivariate normal")
        self.test_initialization(init_param_multivariate_normal)

        print("---------------- \n Testing out values with uniform")
        self.test_initialization(init_param_uniform)

    def changing_stepsize(self) -> None:
        # for constant stepsize, use ETA
        eta_vec = np.array([i * 0.1 for i in range(1, 30)])

        x_star, grad_sgd_loss = self.estimate_x_star()
        for eta_val in tqdm(eta_vec):
            x_iterate, x_iterate_average = self.estimate_x_tilda_k(
                grad_sgd_loss,
                self.B,
                self.N,
                eta_val,
                10,
                x_star,
            )

    def decreasing_stepsize(self) -> None:
        # For decreasing stepsize, use ETA_0, ALPHA
        eta_0_vec = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        alpha = np.array([0.5 + np.random.uniform(0, 0.5) for i in range(10)])
        print(f"eta_0_vec is {eta_0_vec}")
        print(f"alpha is {alpha}")
        x_star, grad_sgd_loss = self.estimate_x_star()

        for eta_0_val in tqdm(eta_0_vec):
            for alpha_val in tqdm(alpha):
                x_iterate, x_iterate_average = self.estimate_x_tilda_k(
                    grad_sgd_loss,
                    self.B,
                    self.N,
                    eta_0_val,
                    10,
                    x_star,
                    alpha_val,
                )

    def changing_gradient_noise(self) -> None:
        # change B to see its effect on the algorithm
        B_vec = np.array([i * 10 for i in range(1, 20)])
        x_star, grad_sgd_loss = self.estimate_x_star()
        for B_val in tqdm(B_vec):
            x_iterate, x_iterate_average = self.estimate_x_tilda_k(
                grad_sgd_loss,
                B_val,
                self.N,
                self.ETA,
                10,
                x_star,
            )

    def changing_loss(self) -> None:
        nu_vec = np.array([10, 50, 100, 200, 300, 400, 500, 1000, 2000])
        nu_inf = np.array([np.inf])

        generate_seed = random.randint(0, 100)
        true_beta, Y, Z = generate_data(self.N, self.D, generate_seed)
        init_param = np.zeros(self.D + 1)

        for nu_val in tqdm(nu_vec):
            sgd_loss, grad_sgd_loss = make_sgd_robust_loss(Y, Z, nu_val)
            res_minimize = sp.optimize.minimize(
                sgd_loss,
                init_param,
                args=(np.arange(self.N),),
            )
            x_star = res_minimize.x
            x_iterate, x_iterate_average = self.estimate_x_tilda_k(
                grad_sgd_loss,
                self.B,
                self.N,
                self.ETA,
                10,
                x_star,
            )
        sgd_loss, grad_sgd_loss = make_sgd_robust_loss(Y, Z, np.inf)
        print("Infinite nu")
        res_minimize = sp.optimize.minimize(
            sgd_loss,
            init_param,
            args=(np.arange(self.N),),
        )
        x_star = res_minimize.x
        x_iterate, x_iterate_average = self.estimate_x_tilda_k(
            grad_sgd_loss,
            self.B,
            self.N,
            self.ETA,
            10,
            x_star,
        )


"""
TODO: function for stats
TODO: function for plotting 
TODO: function for getting x_*
TODO: function for changing batchsize
"""


def main():
    experiments = run_experiments(N, D, NU, ETA, ETA_0, ALPHA, B)
    # experiments.init_param_test()
    # experiments.find_best_num_epochs()
    # experiments.find_best_initilization_param()
    # experiments.changing_stepsize_initstepsize_decayrate()
    # experiments.decreasing_stepsize()
    experiments.changing_loss()


main()
