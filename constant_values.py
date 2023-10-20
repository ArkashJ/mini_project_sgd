def init_constants() -> (int, float, float, float, float, int, int, int):
    NU: int = 5
    ETA: float = 0.2
    ETA_0: float = 5
    ALPHA: float = 0.51
    B: int = 10
    N: int = 10000
    D: int = 10

    print(
        f"Batch size: {B}\n"
        f"Number of samples: {N}\n"
        f"Number of features: {D}\n"
        f"Degrees of freedom: {NU}\n"
        f"Initial step size: {ETA_0}\n"
        f"Decay rate: {ALPHA}\n"
        f"step size: {ETA}\n"
    )

    return NU, ETA, ETA_0, ALPHA, B, N, D
