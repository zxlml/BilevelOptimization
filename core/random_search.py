import torch
from .utils import infinite_loader, score_on_dataset
from .ablo import sgd_step
import math
import logging

def train(tr_dataset, val_dataset, te_dataset, loss_cls, lamb_gen, K, lr_l, wd_l, tr_batch_size, eval_batch_size,
          device, record, print_every=50):
    r"""Random-search baseline: sample lamb, train theta from scratch (K inner
    steps), keep the lamb with the best full-dataset validation loss."""
    train_dataset_loader = infinite_loader(tr_dataset, batch_size=tr_batch_size)

    best_loss_val = float('inf')
    best_lamb = None
    best_theta = None
    for it_h, lamb in enumerate(lamb_gen):
        theta = loss_cls.init_theta(requires_grad=False)
        for it_l in range(K):
            z_tr = [item.to(device) for item in next(train_dataset_loader)]
            theta, loss_tr_sgd = sgd_step(theta, lamb, loss_cls.loss_in, z_tr, lr_l, wd_l)
            theta = [item.detach() for item in theta] if isinstance(theta, list) else theta.detach()
        with torch.no_grad():
            loss_val = score_on_dataset(val_dataset, lambda z: loss_cls.loss_out(lamb, theta, z), eval_batch_size).item()
        if loss_val < best_loss_val or it_h == 0:
            best_loss_val = loss_val
            best_lamb = lamb
            best_theta = [item.detach() for item in theta] if isinstance(theta, list) else theta.detach()

        if it_h % print_every == 0:
            logging.info("it_h: {}\tbest_loss_val: {:.6f}".format(it_h, best_loss_val))
        record["loss_tr_sgd"].append((it_h, loss_tr_sgd))
        record["loss_val_batch"].append((it_h, loss_val))
        with torch.no_grad():
            loss_te = score_on_dataset(te_dataset, lambda z: loss_cls.loss_out(best_lamb, best_theta, z), eval_batch_size).item()
            zero_one_loss_val = score_on_dataset(val_dataset, lambda z: loss_cls.zero_one_loss(best_lamb, best_theta, z), eval_batch_size).item()
            zero_one_loss_te = score_on_dataset(te_dataset, lambda z: loss_cls.zero_one_loss(best_lamb, best_theta, z), eval_batch_size).item()
        record["loss_val"].append((it_h, best_loss_val))
        record["zero_one_loss_val"].append((it_h, zero_one_loss_val))
        record["loss_te"].append((it_h, loss_te))
        record["zero_one_loss_te"].append((it_h, zero_one_loss_te))
        record["gap"].append((it_h, loss_te - best_loss_val))
        record["zero_one_gap"].append((it_h, zero_one_loss_te - zero_one_loss_val))

        if math.isnan(best_loss_val):
            logging.info('nan at {}'.format(it_h))
            break
