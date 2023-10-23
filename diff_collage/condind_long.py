import torch as th
from einops import rearrange

from .base_worker import BaseWorker
from .w_img import split_wimg, avg_merge_wimg

class CondIndLong(BaseWorker):
    """
    Leaverage conditional independence to generate long images
    """
    def __init__(self, shape, eps_scalar_t_fn, num_img, overlap_size=32, sigma_max=80.0, sigma_min=1e-3):
        c, h, w = shape
        assert overlap_size == w // 2
        self.overlap_size = overlap_size
        self.num_img = num_img
        final_img_w = w * num_img - self.overlap_size * (num_img - 1)
        super().__init__((c, h, final_img_w), self.get_eps_t_fn(eps_scalar_t_fn), sigma_min=sigma_min, sigma_max=sigma_max)

    def loss(self, x):
        x1, x2 = x[:-1], x[1:]
        return th.sum(
            (th.abs(x1[:, :, :, -self.overlap_size :] - x2[:, :, :, : self.overlap_size])) ** 2,
            dim=(1, 2, 3),
        )

    def get_eps_t_fn(self, eps_scalar_t_fn):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            xs = split_wimg(long_x, self.num_img, rtn_overlap=False)
            full_eps = eps_scalar_t_fn(xs, scalar_t, enable_grad) # #((b,n), c, h, w)
            full_eps = rearrange(
                full_eps,
                "(b n) c h w -> n b c h w", n = self.num_img
            )

            # calculate half eps
            half_eps = eps_scalar_t_fn(xs[:,:,:,-self.overlap_size:], scalar_t, enable_grad) #((b,n), c, h, w//2)
            half_eps = rearrange(
                half_eps,
                "(b n) c h w -> n b c h w", n = self.num_img
            )

            half_eps[-1]=0

            full_eps[:,:,:,:,-self.overlap_size:] = full_eps[:,:,:,:,-self.overlap_size:] - half_eps
            whole_eps = rearrange(
                full_eps,
                "n b c h w -> (b n) c h w"
            )
            return avg_merge_wimg(whole_eps, self.overlap_size, n=self.num_img, is_avg=False)
        return eps_t_fn
