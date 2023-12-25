#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/4/25 19:06
# @Author : 慕夏
# @File : optimizer.py
# @Project :HyperKG

import math
import torch
from torch.optim.optimizer import Optimizer
from utils.ponicare_utils import *
from utils.euclidean import *

def euclidean_update(p, d_p, lr):
    p.data = p.data - lr * d_p
    return p.data


def poincare_grad(p, d_p):
    p_sqnorm = torch.clamp(torch.sum(p.data ** 2, dim=-1, keepdim=True), 0, 1 - 1e-5)
    d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p


def poincare_update(p, d_p, lr):
    v = -lr * d_p
    p.data = full_p_exp_map(p.data, v)
    return p.data


class RiemannianSGD(Optimizer):

    def __init__(self, params, lr=0.1, param_names=[]):
        defaults = dict(lr=lr)
        super(RiemannianSGD, self).__init__(params, defaults)
        self.param_names = param_names

    def step(self, lr=None):
        loss = None
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group["lr"]
                if self.param_names[i] in ["Eh.weight", "rvh.weight"]:
                    d_p = poincare_grad(p, d_p)
                    p.data = poincare_update(p, d_p, lr)
                else:
                    p.data = euclidean_update(p, d_p, lr)
        return loss



class RiemannianAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super(RiemannianAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.euclidean_manifold = Euclidean()

    def reset_value(self, target, source):
        return target.copy_(source) if target.stride() != source.stride() else target.set_(source)

    def step(self, closure=None):
        loss = closure() if closure else None
        with torch.no_grad():
            for p_group in self.param_groups:
                if "step" not in p_group:
                    p_group["step"] = 0
                betas = p_group["betas"]
                weight_decay = p_group["weight_decay"]
                eps = p_group["eps"]
                learning_rate = p_group["lr"]
                amsgrad = p_group["amsgrad"]

                for point in p_group["params"]:
                    if isinstance(point, (ManifoldParameter)):
                        manifold = point.manifold
                        c = point.c
                    else:
                        manifold = self.euclidean_manifold
                        c = None
                    grad = point.grad
                    if grad is None:
                        continue

                    state = self.state[point]
                    if len(state) == 0:
                        if amsgrad:
                            state["max_exp_avg_sq"] = torch.zeros(point.size(), dtype=point.dtype, device=point.device)
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros(point.size(), dtype=point.dtype, device=point.device)
                        state["exp_avg_sq"] = torch.zeros(point.size(), dtype=point.dtype, device=point.device)

                    grad.add_(weight_decay * point)
                    grad = manifold.egrad2rgrad(point, grad, c)
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(betas[0]).add_((1 - betas[0]) * grad)
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(betas[1]).add_(
                        (1 - betas[1]) * manifold.inner(point, c, grad, keepdim=True)
                    )
                    if amsgrad:
                        max_exp_avg_sq = torch.max(state["max_exp_avg_sq"], exp_avg_sq)
                        denom = torch.add(max_exp_avg_sq.sqrt(), eps)
                    else:
                        denom = torch.add(exp_avg_sq.sqrt(), eps)
                    p_group["step"] += 1
                    bias_cor1 = 1 - math.pow(betas[0], p_group["step"])
                    bias_cor2 = 1 - math.pow(betas[1], p_group["step"])
                    step_size = (
                            learning_rate * bias_cor2 ** 0.5 / bias_cor1
                    )
                    direction = exp_avg / denom
                    new_point = manifold.proj(manifold.expmap(-step_size * direction, point, c), c)
                    exp_avg_new = manifold.ptransp(point, new_point, exp_avg, c)
                    self.reset_value(point, new_point)
                    exp_avg.set_(exp_avg_new)
                    p_group["step"] += 1

        return loss