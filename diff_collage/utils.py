import torch as th
from collections import defaultdict
from tqdm import tqdm
from typing import Callable
import math

def sampling(  # pylint: disable=too-many-locals
    x: th.Tensor,
    rev_ts: th.Tensor,
    noise_fn: Callable,
    x0_pred_fn: Callable,
    s_churn: float = 0.0,
    before_step_fn: Callable = None,
    is_tqdm: bool = True,
    return_traj: bool = True,
):
    """

    reproduce edm sampling

    :param rev_ts: time steps in reverse order for sampling
    :type rev_ts: th.Tensor
    :param noise_fn: a function that generates noise
    :type noise_fn: Callable
    :param x0_pred_fn: x0 prediction function based on noisy data xt and noise level t
    :type x0_pred_fn: Callable
    :param s_churn: how much stochasticity injected, defaults to 0.0
    :type s_churn: float, optional
    :param before_step_fn: a callack function in from of each step, defaults to None
    :type before_step_fn: Callable, optional
    :param is_tqdm: whether open tqdm, defaults to True
    :type is_tqdm: bool, optional
    :param return_traj: return whole sampling traj or just final sample, defaults to True
    :type return_traj: bool, optional
    :return: return final sample or whole traj
    """
    measure_loss = defaultdict(list)
    traj = defaultdict(list)
    if callable(x):
        x = x()
    if traj:
        traj["xt"].append(x.cpu())

    s_t_min = 0.05
    s_t_max = 50.0
    s_noise = 1.003
    eta = min(s_churn / len(rev_ts), math.sqrt(2.0) - 1)

    loop = zip(rev_ts[:-1], rev_ts[1:])
    if is_tqdm:
        loop = tqdm(loop)

    running_x = x
    for cur_t, next_t in loop:
        # cur_x = traj["xt"][-1].clone().to("cuda")
        cur_x = running_x
        if cur_t < s_t_max and cur_t > s_t_min:
            hat_cur_t = cur_t + eta * cur_t
            cur_noise = noise_fn(cur_x, cur_t)
            cur_x = cur_x + s_noise * cur_noise * th.sqrt(hat_cur_t ** 2 - cur_t ** 2)
            cur_t = hat_cur_t

        if before_step_fn is not None:
            cur_x = before_step_fn(cur_x, cur_t)

        x0, loss_info, traj_info = x0_pred_fn(cur_x, cur_t)
        epsilon_1 = (cur_x - x0) / cur_t

        xt_next = x0 + next_t * epsilon_1

        x0, loss_info, traj_info = x0_pred_fn(xt_next, next_t)
        epsilon_2 = (xt_next - x0) / next_t

        xt_next = cur_x + (next_t - cur_t) * (epsilon_1 + epsilon_2) / 2

        running_x = xt_next

        if return_traj:
            for key, value in loss_info.items():
                measure_loss[key].append(value)

            for key, value in traj_info.items():
                traj[key].append(value)
            traj["xt"].append(running_x.to("cpu").detach())

    if return_traj:
        return traj, measure_loss
    return running_x
