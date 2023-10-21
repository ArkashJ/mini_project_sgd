from autograd import grad
import autograd.numpy as np
from sgd_robust_regression import *
import matplotlib.pyplot as plt
import seaborn as sns

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


def main():
    experiments = run_experiments(N, D, NU, ETA, ETA_0, ALPHA, B)
    experiments.init_param_test()


main()
