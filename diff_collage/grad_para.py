import torch as th
from einops import rearrange

from .base_worker import BaseWorker, batch_mul
from .w_img import split_wimg, avg_merge_wimg
from .loss_helper import get_x0_grad_pred_fn


class ParaGradCor(BaseWorker):
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

    def loss(self, x):
        x1, x2 = x[:-1], x[1:]
        return th.sum(
            (th.abs(x1[:, :, :, -self.overlap_size :] - x2[:, :, :, : self.overlap_size])) ** 2,
            dim=(1, 2, 3),
        )

    def generate_xT(self, n):
        white_noise = th.randn((n , *self.shape)).cuda()
        return self.noise(white_noise, None) * 80.0


    def x0_fn(self, xt, scalar_t, enable_grad=False):
        cur_eps = self.eps_scalar_t_fn(xt, scalar_t, enable_grad)
        return xt - scalar_t * cur_eps, {}, {}

    def noise(self, xt, scalar_t):
        del scalar_t
        noise = th.randn_like(xt)
        b, _, _, w = xt.shape
        final_img_w = w * b - self.overlap_size * (b - 1)
        noise = rearrange(noise, "(t n) c h w -> t c h (n w)", t=1)[:, :, :, :final_img_w]
        noise = split_wimg(noise, b, rtn_overlap=False)
        return noise

    def optimal_weight_fn(self, xs, grads, *args):
        del args
        overlap_size = self.overlap_size
        # argmin_{w} (delta_pixel - w * delta_pixel)^2
        delta_pixel = xs[:-1, :, :, -overlap_size:] - xs[1:, :, :, :overlap_size]
        delta_grads = grads[:-1, :, :, -overlap_size:] - grads[1:, :, :, :overlap_size]
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