import torch as th

from .base_worker import BaseWorker, batch_mul

class ReplaceInp(BaseWorker):
    def __init__(self, x0, mask, eps_scalar_t_fn):
        self.given_x0 = x0.cuda()
        self.mask = mask.cuda()
        super().__init__(None, self.get_eps_fn(eps_scalar_t_fn))

    def generate_xT(self, n):
        del n
        return 80.0 * th.randn_like(self.given_x0)

    def get_eps_fn(self, eps_scalar_t_fn):
        def eps_fn(x, scalar_t, require_grad):
            eps_pred = eps_scalar_t_fn(x, scalar_t, require_grad)
            x0_pred = x - scalar_t * eps_pred
            x0_replace = th.where(
                self.mask.unsqueeze(1) > 0,
                self.given_x0,
                x0_pred
            )
            return (x - x0_replace) / scalar_t

        return eps_fn

def get_x0_grad_fn(raw_x0_pred_fn, cond_loss_fn, weight_fn, x0_fn, thres_t):
    def fn(xt, scalar_t):
        xt = xt.requires_grad_(True)
        x0_pred = raw_x0_pred_fn(xt, scalar_t)

        loss_info = {
            "raw_x0": cond_loss_fn(x0_pred.detach()).cpu(),
        }

        traj_info = {
            "t": scalar_t,
        }
        if scalar_t < thres_t:
            x0_cor = x0_pred.detach()
        else:
            pred_loss = cond_loss_fn(x0_pred)
            grad_term = th.autograd.grad(pred_loss.sum(), xt)[0]
            weights = weight_fn(x0_pred, grad_term, cond_loss_fn)
            x0_cor = (x0_pred - batch_mul(weights, grad_term)).detach()
            loss_info["weight"] = weights.detach().cpu()
            traj_info["grad"] = grad_term.detach().cpu()

        if x0_fn:
            x0 = x0_fn(x0_cor, scalar_t)
        else:
            x0 = x0_cor

        loss_info["cor_x0"] = cond_loss_fn(x0_cor.detach()).cpu()
        loss_info["x0"] = cond_loss_fn(x0.detach()).cpu()
        traj_info.update({
                "raw_x0": x0_pred.detach().cpu(),
                "cor_x0": x0_cor.detach().cpu(),
                "x0": x0.detach().cpu(),
            }
        )
        return x0_cor, loss_info, traj_info

    return fn

class GradInp(BaseWorker):
    def __init__(self, given_x, mask, eps_scalar_t_fn):
        self.given_x0 = given_x.cuda()
        self.mask = mask.cuda()

        super().__init__(None, eps_scalar_t_fn)


    def generate_xT(self, n):
        del n
        return 80.0 * th.randn_like(self.given_x0)

    def delta_diff(self, x0_pred):
        return (x0_pred - self.given_x0) * self.mask.unsqueeze(1)

    def cond_loss_fn(self, x0_pred):
        return th.sum(
            (self.delta_diff(x0_pred)) ** 2,
            dim=(1, 2, 3),
        )

    def raw_x0_pred_fn(self, xt, scalar_t):
        cur_eps = self.eps_scalar_t_fn(xt, scalar_t, True)
        x0 = xt - scalar_t * cur_eps
        return x0

    def x0_replace(self, x0, *args):
        rtn_x0 = th.where(
                self.mask.unsqueeze(1) > 0,
                self.given_x0,
                x0,
            )
        return rtn_x0

    def optimal_weight_fn(self, x0, grads, *args, ratio=1.0):
        del args
        # argmin_{w} (delta_pixel - w * delta_pixel)^2
        delta_pixel = self.delta_diff(x0)
        
        delta_grads = grads * self.mask.unsqueeze(1)

        num = th.sum(delta_pixel * delta_grads, dim=[1,2,3])
        denum = th.sum(delta_grads * delta_grads, dim=[1,2,3])
        _optimal_weight = num / denum
        # if math.isnan(_optimal_weight):
        if th.isnan(_optimal_weight).any():
            print(denum)
            raise RuntimeError("nan for weights")

        return ratio * _optimal_weight * th.ones(x0.shape[0]).to(x0)
