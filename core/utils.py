from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import os
import csv
import pprint
import logging
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def infinite_loader(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    while True:
        for data in loader:
            yield data


def score_on_dataset(dataset, score_fn, batch_size):
    r"""
    Args:
        dataset: an instance of Dataset
        score_fn: a batch of data -> a batch of scalars
        batch_size: the batch size
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    total_score = 0.
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for z in dataloader:
        z = [item.to(device) for item in z]
        score = score_fn(z)
        total_score += score.sum().detach()
    mean_score = total_score / len(dataset)
    return mean_score


def backup_args(args, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "args.txt")
    s = pprint.pformat(args)
    with open(path, 'w') as f:
        f.write(s)


def detach(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.detach()
    elif isinstance(inputs, list) or isinstance(inputs, tuple):
        return [item.detach() for item in inputs]
    else:
        raise TypeError


def iter_islast(iterable):
    it = iter(iterable)
    prev = next(it)
    for item in it:
        yield False, prev
        prev = item
    yield True, prev


def plot_record(record, path):
    os.makedirs(path, exist_ok=True)
    for curve_name, data in record.items():
        if len(data) == 0:
            continue
        x = list(map(lambda z: z[0], data))
        y = list(map(lambda z: z[1], data))
        plt.plot(x, y, label="{}".format(curve_name))
        plt.title(curve_name)
        plt.savefig(os.path.join(path, "%s.png" % curve_name))
        plt.close()


def save_record_csv(record, path):
    r"""Dump a record dict {(it, val) lists} to CSV, one row per iteration."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = {}
    for key, data in record.items():
        for it, val in data:
            rows.setdefault(it, {})[key] = val
    its = sorted(rows.keys())
    keys = list(record.keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration"] + keys)
        for it in its:
            writer.writerow([it] + [rows[it].get(k, "") for k in keys])


def average_records(records):
    r"""Average record dicts (same keys, same evaluation grids) into mean/std."""
    mean_rec, std_rec = {}, {}
    for key in records[0]:
        grids = [dict(rec[key]) for rec in records]
        its = sorted(grids[0].keys())
        assert all(sorted(g.keys()) == its for g in grids), "eval grids differ across runs"
        vals = np.array([[g[i] for i in its] for g in grids])
        mean_rec[key] = list(zip(its, vals.mean(axis=0)))
        std_rec[key] = list(zip(its, vals.std(axis=0)))
    return mean_rec, std_rec


def set_logger(fname):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(fname, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
