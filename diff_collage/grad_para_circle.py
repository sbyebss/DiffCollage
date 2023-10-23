import torch as th
from einops import rearrange

from .base_worker import BaseWorker, batch_mul
from .w_img import split_wimg, avg_merge_wimg
from .loss_helper import get_x0_grad_pred_fn
import numpy as np


class ParaGradCorCircle(BaseWorker):
    def __init__(self, shape, eps_scalar_t_fn, overlap_size, thres_t=1.0, adam_num_iter=100):
        super().__init__(shape, eps_scalar_t_fn)
        self.overlap_size = overlap_size
        self.adam_num_iter = adam_num_iter

        self.grad_x0_fn = get_x0_grad_pred_fn(
            lambda _x,_t: self.x0_fn(_x, _t, enable_grad=True)[0],
            self.loss,
            self.adam_grad_weight,
            x0_update=None,
            thres_t=thres_t,
        )

    def get_match_patch(self, x):
        tail = x[:, :, :, -self.overlap_size :]
        head = x[:, :, :, : self.overlap_size]
        tail = th.roll(tail, 1, 0)
        return tail, head

    def loss(self, x):
        tail, head = self.get_match_patch(x)
        return th.sum(
            (tail - head)**2,
            dim=(1, 2, 3),
        )

    def generate_xT(self, n):
        noise = th.randn((n , *self.shape)).cuda()
        split_noise = self.noise(noise, None) * 80.0
        return split_noise

    def noise(self, xt, cur_t):
        del cur_t
        noise = th.randn_like(xt)
        b, _, _, w = xt.shape
        final_img_w = w * b - self.overlap_size * b
        noise = rearrange(noise, "(t n) c h w -> t c h (n w)", t=1)[:, :, :, :final_img_w]
        noise = th.cat([noise, noise[:,:,:, :self.overlap_size]], dim=-1)
        noise, _ = split_wimg(noise, b)
        return noise

    def merge_circle_image(self, xt):
        merged_long_img = avg_merge_wimg(xt, self.overlap_size)
        return th.cat(
            [
                (merged_long_img[:,:,:self.overlap_size] + merged_long_img[:,:,-self.overlap_size:]) / 2.0,
                merged_long_img[:,:,self.overlap_size:-self.overlap_size],
            ],
            dim=-1
        )

    def split_circle_image(self, merged_long_img, n):
        imgs,_ = split_wimg(
            th.cat(
                [
                    merged_long_img,
                    merged_long_img[:,:,:self.overlap_size],
                ],
                dim = -1,
            ),
            n
        )
        return imgs


    def optimal_weight_fn(self, xs, grads, *args):
        del args
        # argmin_{w} (delta_pixel - w * delta_pixel)^2
        tail, head = self.get_match_patch(xs)
        delta_pixel = tail - head
        tail, head = self.get_match_patch(grads)
        delta_grads = tail - head

        num = th.sum(delta_pixel * delta_grads).item()
        denum = th.sum(delta_grads * delta_grads).item()
        _optimal_weight = num / denum
        return _optimal_weight * th.ones(xs.shape[0]).to(xs)

    def adam_grad_weight(self, x0, grad_term, cond_loss_fn):
        init_weight = self.optimal_weight_fn(x0, grad_term)
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

    # TODO:
    def x0_replace(self, x0, sclar_t, thres_t):
        if sclar_t > thres_t:
            merge_x0 = avg_merge_wimg(x0, self.overlap_size)
            return split_wimg(merge_x0, x0.shape[0])[0]
        else:
            return x0

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