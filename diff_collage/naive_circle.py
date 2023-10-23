import torch as th
from einops import rearrange

from .base_worker import BaseWorker
import numpy as np


class NaiveCircle(BaseWorker):
    def merge_circle_image(self, x):
        del self
        return rearrange(x, "b c h w -> c h (b w)")

    def split_circle_image(self, img, b):
        del self
        return rearrange(img, "c h (b w) -> b c h w", b=b)

    def before_step_fn(self, x, t):
        del t
        merge_img = self.merge_circle_image(x)
        idx = np.random.randint(merge_img.shape[-1] - 3)
        merge_img = th.cat(
            [
                merge_img[:,:,idx:],
                merge_img[:,:,:idx],
            ],
            dim=-1
        )
        return self.split_circle_image(merge_img, x.shape[0])