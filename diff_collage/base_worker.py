import numpy as np
import torch as th

__all__ = [
    "generic_sampler",
    "BaseWorker",
]


def batch_mul(a, b):  # pylint: disable=invalid-name
    return th.einsum("a...,a...->a...", a, b)

class BaseWorker:
    """
    Some utility functions for EDM sampling
    """
    def __init__(self, shape, eps_scalar_t_fn, sigma_max=80.0, sigma_min=1e-3):
        self.shape = shape
        self.eps_scalar_t_fn = eps_scalar_t_fn
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def generate_xT(self, n):
        return self.sigma_max * th.randn((n , *self.shape)).cuda()

    def x0_fn(self, xt, scalar_t, enable_grad=False):
        cur_eps = self.eps_scalar_t_fn(xt, scalar_t, enable_grad)
        x0 = xt - scalar_t * cur_eps
        x0 = th.clip(x0, -1,1) # static thresholding algorithm
        return x0, {}, {"x0": x0.cpu()}

    def noise(self, xt, scalar_t):
        del scalar_t
        return th.randn_like(xt)

    def rev_ts(self, n_step, ts_order):
        _rev_ts = th.pow(
            th.linspace(
                np.power(self.sigma_max, 1.0 / ts_order),
                np.power(self.sigma_min, 1.0 / ts_order),
                n_step + 1
            ),
            ts_order
        )
        return _rev_ts.cuda()