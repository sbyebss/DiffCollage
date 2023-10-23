import torch as th
from einops import rearrange

from .base_worker import BaseWorker


class NaiveOccLong(BaseWorker):
    def merge(self, x):
        del self
        return rearrange(x, "b c h w -> c h (b w)")

    def split(self, img, b):
        del self
        return rearrange(img, "c h (b w) -> b c h w", b=b)

    def x0_fn(self, xt, scalar_t, enable_grad=False):
        batch_size = xt.shape[0]
        merge_img = self.merge(xt)
        mid_xt = self.split(merge_img[:,:,32:-32], batch_size -1)
        x0, _, _ = super().x0_fn(
            th.cat([xt, mid_xt]), 
            scalar_t, enable_grad
        )

        full_x0 = self.merge(x0[:batch_size])
        mid_x0 = self.merge(x0[batch_size:])
        x0 = th.cat([
            full_x0[:,:,:32],
            (full_x0[:,:,32:-32] + mid_x0) / 2.0,
            full_x0[:,:,-32:]
        ], dim=-1)
        return self.split(x0, batch_size), {}, {}
