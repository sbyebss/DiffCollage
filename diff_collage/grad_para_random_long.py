import torch as th
from einops import rearrange

from .base_worker import BaseWorker, batch_mul
from .loss_helper import get_x0_grad_pred_fn
import numpy as np


class SplitMergeOp:
    def __init__(self, avg_overlap=32):
        self.avg_overlap = avg_overlap
        self.cur_overlap_int = None

    def sample(self, n):
        # lower_coef = 3 / 4.0
        _lower_bound = self.avg_overlap - 4
        base_overlap = np.ones(n) * _lower_bound

        total_ball = (self.avg_overlap - _lower_bound) * n
        random_number = np.random.randint(0, total_ball - n, n-1)
        random_number = np.sort(random_number)
        balls = np.append(random_number, total_ball - n) - np.insert(random_number, 0, 0) + np.ones(n) + base_overlap

        assert np.sum(balls) == n * self.avg_overlap

        return balls.astype(np.int)

    def reset(self, n):
        self.cur_overlap_int = self.sample(n)

    def split(self, img, n, img_w=64):
        assert img.ndim == 3
        # assert img.shape[-1] > (n-1) * self.avg_overlap
        assert (n-1) == self.cur_overlap_int.shape[0]

        assert (n-1) * self.avg_overlap + img.shape[-1] == n * img_w

        cur_idx = 0
        imgs = []
        for cur_overlap in self.cur_overlap_int:
            imgs.append(img[:,:,cur_idx:cur_idx + img_w])
            cur_idx = cur_idx + img_w - cur_overlap
        imgs.append(img[:,:,cur_idx:])
        return th.stack(imgs)

    def merge(self, imgs):
        b = imgs.shape[0]
        img_size = imgs.shape[-1]
        assert b - 1 == self.cur_overlap_int.shape[0]
        img_width = b * imgs.shape[-1] - np.sum(self.cur_overlap_int)
        wimg = th.zeros((3, imgs.shape[-2], img_width)).to(imgs)
        ncnt = th.zeros(img_width).to(imgs)
        cur_idx = 0
        for i_th, cur_img in enumerate(imgs):
            wimg[:,:,cur_idx:cur_idx + img_size] += cur_img
            ncnt[cur_idx:cur_idx + img_size] += 1.0
            if i_th < b -1:
                cur_idx = cur_idx + img_size - self.cur_overlap_int[i_th]
        return wimg / ncnt[None,None,:]


class ParaGradCorRandomLong(BaseWorker):
    def __init__(self, shape, eps_scalar_t_fn, overlap_size, thres_t=1.0, adam_num_iter=100):
        super().__init__(shape, eps_scalar_t_fn)
        self.overlap_size = overlap_size
        self.adam_num_iter = adam_num_iter
        self.op = SplitMergeOp(overlap_size)

        self.grad_x0_fn = get_x0_grad_pred_fn(
            lambda _x,_t: self.x0_fn(_x, _t, enable_grad=True)[0],
            self.loss,
            self.adam_grad_weight,
            x0_update=None,
            thres_t=thres_t,
            # x0_update =lambda _x,_t: self.x0_replace(_x, _t, thres_t),
        )

    def loss(self, x):
        avg_x = self.op.split(
            self.op.merge(x), x.shape[0], x.shape[-1]
        )
        return th.sum(
            (x - avg_x) ** 2,
            dim=(1, 2, 3),
        )

    def x0_fn(self, xt, scalar_t, enable_grad=False):
        cur_eps = self.eps_scalar_t_fn(xt, scalar_t, enable_grad)
        return xt - scalar_t * cur_eps, {}, {}

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

    def adam_grad_weight(self, x0, grad_term, cond_loss_fn):
        init_weight = th.ones(x0.shape[0]).to(x0)
        grad_term = grad_term.detach()
        x0 = x0.detach()
        with th.enable_grad():
            weights = init_weight.requires_grad_()
            optimizer = th.optim.Adam(
                [
                    weights,
                ],
                lr=1e-2,
            )

            def _loss(w):
                cor_x0 = x0 - batch_mul(w, grad_term)
                return cond_loss_fn(cor_x0).sum()

            for _ in range(self.adam_num_iter):
                optimizer.zero_grad()
                _cur_loss = _loss(weights)
                _cur_loss.backward()
                optimizer.step()
        return weights

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