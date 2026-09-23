# Core solvers for "Fine-grained Analysis of Stability and Generalization for
# Stochastic Bilevel Optimization" (IJCAI 2024).
#
# Paper convention (gen_sbo.pdf) is used throughout:
#   K = number of outer iterations, T = number of inner iterations per outer loop.
#   - SSGD (Algorithm 1): single timescale, T = 1.
#   - TSGD (Algorithm 2): T inner steps per outer step with warm start
#     y0_{k+1} = y^T_k (NO re-initialization, the key difference from UD).
#   - UD   (Algorithm 3, Bao et al. 2021): same as TSGD but the inner-level
#     parameters are re-initialized before each outer loop (re_init=True).
import math
import logging

import torch
import torch.autograd as autograd
import torch.optim as optim

import func as func
from .utils import infinite_loader, score_on_dataset


def sgd_step(theta, lamb, loss, z, lr, wd):
    r"""
    One inner-loop SGD step on theta, unrolled so that gradients keep flowing
    to lamb (the outer-level variable) through create_graph=True.
    :return: (1 - alpha * wd) * theta - alpha * D_theta loss(lamb, theta, z), loss value
    """
    with func.RequiresGradContext(theta, requires_grad=True):
        loss_ = loss(lamb, theta, z).mean()
        g = autograd.grad(loss_, theta, create_graph=True)
    if isinstance(theta, list) or isinstance(theta, tuple):
        return [(1. - lr * wd) * a - lr * b for a, b in zip(theta, g)], loss_.item()
    else:
        return (1. - lr * wd) * theta - lr * g[0], loss_.item()


@torch.no_grad()
def evaluate(loss_cls, lamb, theta, val_dataset, te_dataset, eval_batch_size, record, it):
    r"""Full-dataset evaluation. The generalization gap (Eq. 2) is estimated by
    the divergence between the validation error and the testing error, both
    computed on the FULL validation / test sets (paper Sec. 5.2)."""
    lamb = [lamb] if isinstance(lamb, torch.Tensor) else lamb
    loss_val = score_on_dataset(val_dataset, lambda z: loss_cls.loss_out(lamb, theta, z), eval_batch_size).item()
    loss_te = score_on_dataset(te_dataset, lambda z: loss_cls.loss_out(lamb, theta, z), eval_batch_size).item()
    zo_val = score_on_dataset(val_dataset, lambda z: loss_cls.zero_one_loss(lamb, theta, z), eval_batch_size).item()
    zo_te = score_on_dataset(te_dataset, lambda z: loss_cls.zero_one_loss(lamb, theta, z), eval_batch_size).item()
    record["loss_val"].append((it, loss_val))
    record["loss_te"].append((it, loss_te))
    record["zero_one_loss_val"].append((it, zo_val))
    record["zero_one_loss_te"].append((it, zo_te))
    record["gap"].append((it, loss_te - loss_val))
    record["zero_one_gap"].append((it, zo_te - zo_val))
    return loss_val, loss_te


def train(tr_dataset, val_dataset, te_dataset, loss_cls, K, T, lr_l, wd_l, lr_h, wd_h, mm_h,
          tr_batch_size, val_batch_size, eval_batch_size, device, record,
          re_init=False, eval_every=50, print_every=200, clip_grad_norm=None):
    r"""
    SSGD / TSGD / UD for the data reweighting task (paper Sec. 5).

    Args:
        K: number of outer iterations (paper convention).
        T: number of inner iterations per outer loop; T = 1 gives SSGD (Alg. 1),
           T > 1 with warm start gives TSGD (Alg. 2), re_init=True gives UD (Alg. 3).
        lr_l, lr_h: inner / outer step sizes (paper: 0.01 and 5).
        re_init: if True, re-initialize theta before each outer loop (UD, Alg. 3);
                 otherwise warm start y0_{k+1} = y^T_k (SSGD/TSGD).
        eval_every: full-dataset evaluation frequency (in outer iterations).

    Returns:
        (lamb, theta): detached final outer/inner parameters, used e.g. for
        measuring the empirical on-average argument stability (Definition 5).
    """
    lamb = loss_cls.init_lamb(requires_grad=True)
    train_dataset_loader = infinite_loader(tr_dataset, batch_size=tr_batch_size)
    val_dataset_loader = infinite_loader(val_dataset, batch_size=val_batch_size)

    opt = optim.SGD([lamb] if isinstance(lamb, torch.Tensor) else lamb, lr=lr_h, weight_decay=wd_h, momentum=mm_h)
    theta0 = loss_cls.init_theta(requires_grad=False)
    theta = [item.clone().detach() for item in theta0]
    for it_h in range(K):
        if re_init:
            # UD (Algorithm 3): re-initialize inner-level parameters to the SAME
            # initial point y0_k = y0 before each outer loop
            theta = [item.clone().detach() for item in theta0]
        else:
            # SSGD/TSGD (Algorithm 1/2): warm start y0_{k+1} = y^T_k, cut the graph
            theta = [item.clone().detach().to(device).requires_grad_(False) for item in theta]

        for it_l in range(T):
            z_tr = [item.to(device) for item in next(train_dataset_loader)]
            theta, loss_tr_sgd = sgd_step(theta, lamb, loss_cls.loss_in, z_tr, lr_l, wd_l)

        # outer step: one minibatch from the validation set, hypergradient flows
        # through the T unrolled inner steps: x_{k+1} = x_k - lr_h * grad_x f(x_k, y^T_k(x_k))
        z_val = [item.to(device) for item in next(val_dataset_loader)]
        loss_val_batch = loss_cls.loss_out(lamb, theta, z_val).mean()

        opt.zero_grad()
        loss_val_batch.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_([lamb] if isinstance(lamb, torch.Tensor) else lamb, clip_grad_norm)
        opt.step()

        record["loss_tr_sgd"].append((it_h, loss_tr_sgd))
        record["loss_val_batch"].append((it_h, loss_val_batch.item()))

        if it_h % eval_every == 0 or it_h == K - 1:
            evaluate(loss_cls, lamb, theta, val_dataset, te_dataset, eval_batch_size, record, it_h)
            logging.info("it_h: {}\tloss_val: {:.6f}\tloss_te: {:.6f}\tgap: {:.6f}".format(
                it_h, record["loss_val"][-1][1], record["loss_te"][-1][1], record["gap"][-1][1]))

        if math.isnan(loss_val_batch.item()) or math.isnan(loss_tr_sgd):
            logging.info('nan at outer iteration {}'.format(it_h))
            break

        if it_h % print_every == 0:
            logging.info("it_h: {}\tloss_val_batch: {:.6f}\tloss_tr_sgd: {:.6f}".format(it_h, loss_val_batch.item(), loss_tr_sgd))

    lamb = lamb.detach().clone()
    theta = [item.detach().clone() for item in theta]
    return lamb, theta


def measure_stability(tr_dataset, val_dataset, te_dataset, loss_cls, K, T, lr_l, wd_l, lr_h, wd_h, mm_h,
                      tr_batch_size, val_batch_size, eval_batch_size, device,
                      perturb_indices, perturb_pool, seed=42):
    r"""Empirical l2 on-average argument stability (Definition 5).

    Runs the algorithm on (D_m1, D_m2) and on (D^(i), D_m2) where the i-th
    validation sample is replaced by z~_i drawn independently from D1, from the
    SAME seed / initialization, and measures ||A(D) - A(D^(i))||_2 over (x, y).

    Args:
        perturb_indices: validation positions to replace (one paired run each).
        perturb_pool: independent draws from D1, z~_i = perturb_pool[k] for the
            k-th entry of perturb_indices (e.g. held-out clean MNIST samples).

    Returns:
        list of dicts: [{"outer_iter", "dx2", "dy2", "dxy2", "replaced_index"}, ...]
        ending with the average over replaced samples.
    """
    from copy import deepcopy

    def run_on(val_data):
        record = {k: [] for k in ["loss_tr_sgd", "loss_val_batch", "loss_val", "loss_te",
                                  "zero_one_loss_val", "zero_one_loss_te", "gap", "zero_one_gap"]}
        # same seed => identical minibatch sampling sequence, so the ONLY
        # difference between paired runs is the single replaced sample
        torch.manual_seed(seed)
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        lamb, theta = train(tr_dataset, val_data, te_dataset, loss_cls, K, T, lr_l, wd_l, lr_h, wd_h, mm_h,
                            tr_batch_size, val_batch_size, eval_batch_size, device, record,
                            re_init=False, eval_every=K + 1, print_every=K + 1)
        return lamb, theta

    def param_vec(lamb, theta):
        parts = [lamb.reshape(-1)] if isinstance(lamb, torch.Tensor) else [l.reshape(-1) for l in lamb]
        parts += [p.reshape(-1) for p in theta]
        return torch.cat(parts)

    base_lamb, base_theta = run_on(val_dataset)
    base_vec = param_vec(base_lamb, base_theta)

    results = []
    for k, i in enumerate(perturb_indices):
        val_pert = deepcopy(val_dataset)
        arr = list(val_pert.array)
        assert k < len(perturb_pool), "perturb_pool too small"
        # D^(i): replace the i-th validation sample with an independent draw
        # z~_i ~ D1 (kept as (x, y) tuples, consistent with clean datasets)
        arr[i] = tuple(perturb_pool[k])
        val_pert.array = arr
        pert_lamb, pert_theta = run_on(val_pert)
        pert_vec = param_vec(pert_lamb, pert_theta)
        dxy2 = ((base_vec - pert_vec) ** 2).sum().item()
        dx2 = ((base_lamb - pert_lamb) ** 2).sum().item()
        dy2 = sum(((a - b) ** 2).sum().item() for a, b in zip(base_theta, pert_theta))
        results.append({"outer_iter": K, "dx2": dx2, "dy2": dy2, "dxy2": dxy2, "replaced_index": i})
        logging.info("stability: replaced val[{}]\tdx2={:.3e}\tdy2={:.3e}".format(i, dx2, dy2))
    # average over replaced samples => estimate of l2 on-average argument stability
    avg = {"outer_iter": K,
           "dx2": sum(r["dx2"] for r in results) / len(results),
           "dy2": sum(r["dy2"] for r in results) / len(results),
           "dxy2": sum(r["dxy2"] for r in results) / len(results),
           "replaced_index": "avg"}
    results.append(avg)
    return results
