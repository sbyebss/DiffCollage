from .grad_para_random_long import SplitMergeOp
import torch as th
from einops import rearrange

from .base_worker import BaseWorker


class RandomOccLong(BaseWorker):
    def __init__(self, shape, eps_scalar_t_fn, overlap_size):
        super().__init__(shape, eps_scalar_t_fn)
        self.overlap_size = overlap_size
        self.op = SplitMergeOp(overlap_size)

    def x0_fn(self, xt, scalar_t, enable_grad=False):
        raw_x0,_,_ = super().x0_fn(xt, scalar_t, enable_grad)
        x0_merged = self.op.merge(raw_x0)
        return self.op.split(x0_merged, xt.shape[0], xt.shape[-1]), {},{}

    def generate_xT(self, n):
        noise = th.randn((n , *self.shape)).cuda()
        b, _, _, w = noise.shape
        final_img_w = w * b - self.overlap_size * (b - 1)
        merged_noise = rearrange(noise, "(t n) c h w -> t c h (n w)", t=1)[:, :, :, :final_img_w][0]
        self.op.reset(n-1)
        return self.op.split(merged_noise, n, self.shape[-1]) * 80.0

    def noise(self, xt, scalar_t):
        noise = th.randn_like(xt)
        b, _, _, w = xt.shape
        final_img_w = w * b - self.overlap_size * (b - 1)
        noise = rearrange(noise, "(t n) c h w -> t c h (n w)", t=1)[:, :, :, :final_img_w][0]
        noise = self.op.split(noise, b, w)
        return noise

    def x0_replace(self, x0, sclar_t, thres_t):
        if sclar_t > thres_t:
            merge_x0 = self.op.merge(x0)
            return self.op.split(merge_x0, x0.shape[0], x0.shape[-1])
        else:
            return x0

    def before_step_fn(self, x, t):
        del t
        merge_img = self.op.merge(x)
        self.op.reset(x.shape[0] - 1)
        return self.op.split(merge_img, x.shape[0], x.shape[-1])