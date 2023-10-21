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
        epochs,
        x=np.zeros(D + 1),
        alpha=0,
        avg_range: float = 0.5,
    ) -> Optional[np.ndarray]:
        params = run_sgd(grad_loss, epochs, x, stepsize, alpha, batchsize, n)

        x_iterate_ = params[-1]
         
        k = epochs * n // batchsize
        lower_range, upper_range = math.floor(k * avg_range), k + 1
        x_iterate = 0
            
        # find the iterate average from the last 50% of the iterations


        for i in range(lower_range, upper_range):
            x_iterate += params[i]
        x_iterate_average = x_iterate / (upper_range - lower_range)
        
        return x_iterate_, x_iterate_average

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

    def find_norm(
        self,
        x_star: np.ndarray,
        x_iterate: np.ndarray,
        x_iterate_average: np.ndarray,
    ) -> float:
        print("x-star", x_star)
        temp = []
        print(type(x_iterate))
        # loop through numpy.float64
        difference = x_star - x_iterate
        print(f"difference- \n {difference}")
        for i in range(x_iterate.size):
            val = x_star[i]-x_iterate[i]
            print(f"val- \n {val}")
            val = val**2
            temp.append((x_star[i]-x_iterate[i])**2)
            print(f"val- \n {val}")
        x_iterate = np.array(temp)
        x_iterate = np.sqrt(x_iterate)
        x_iterate = np.linalg.norm(x_star - x_iterate) 
        x_iterate_average = np.linalg.norm(x_star - x_iterate_average)
        print(f"norm_x_star- \n {x_iterate}")
        print(f"norm_x_tilda_k- \n {x_iterate_average}")
        return np.linalg.norm(x_star - x_iterate), np.linalg.norm(
            x_star - x_iterate_average
        )

    def test_norms(self):
        x_star, grad_sgd_loss = self.estimate_x_star()
        # call estimate_x_tilda_k
        x_iterate, x_iterate_average = self.estimate_x_tilda_k(
            grad_sgd_loss,
            self.B,
            self.N,
            self.ETA,
            10,
        )
        print(f"x_iterate- \n {x_iterate}")
        norm_x_star, norm_x_tilda_k = self.find_norm(
            x_star, x_iterate, x_iterate_average
        )

    def find_best_num_epochs(self):
        epoch_increase = 5
        epoch1 = [i for i in range(1, 100, epoch_increase)]
        epoch2 = [200, 300, 400]

        x_star, grad_sgd_loss = self.estimate_x_star()
        store_x_star = [0] * len(epoch1) + [0] * len(epoch2)
        store_x_tilda_k = [0] * len(epoch1) + [0] * len(epoch2)

        for idx, epoch in enumerate(epoch1):
            x_iterate, x_iterate_average = self.estimate_x_tilda_k(
                grad_sgd_loss,
                self.B,
                self.N,
                self.ETA,
                epoch,
            )
            norm_x_star, norm_x_tilda_k = self.find_norm(
                x_star, x_iterate, x_iterate_average
            )
            store_x_star[idx] = norm_x_star
            store_x_tilda_k[idx] = norm_x_tilda_k
        epoch1_len = len(epoch1)
        for idx, epoch in enumerate(epoch2):
            x_iterate, x_iterate_average = self.estimate_x_tilda_k(
                grad_sgd_loss,
                self.B,
                self.N,
                self.ETA,
                epoch,
            )
            norm_x_star, norm_x_tilda_k = self.find_norm(
                x_star, x_iterate, x_iterate_average
            )
            store_x_star[idx + epoch1_len] = norm_x_star
            store_x_tilda_k[idx + epoch1_len] = norm_x_tilda_k
        print(f"store_x_star- \n {store_x_star}")
        print(f"store_x_tilda_k- \n {store_x_tilda_k}")


def main():
    experiments = run_experiments(N, D, NU, ETA, ETA_0, ALPHA, B)
    # experiments.init_param_test()
    # experiments.find_best_num_epochs()
    experiments.test_norms()


main()
