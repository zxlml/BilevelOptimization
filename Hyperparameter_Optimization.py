# Experiment driver for "Fine-grained Analysis of Stability and Generalization
# for Stochastic Bilevel Optimization" (IJCAI 2024) - hyperparameter
# optimization as data reweighting on corrupted MNIST (paper Sec. 5).
#
# Paper convention: K = outer iterations, T = inner iterations (T = 1 -> SSGD).
# Paper settings (Sec. 5.1): 50% corrupted training labels, MLP 784/256/10 with
# cross-entropy, 2000 train / 2000 validation / 1000 test samples, initial
# batch size 8, initial step sizes 0.01 (inner) and 5 (outer), 5 repetitions.
#
#   Fig.1 analog (--mode fig1): vary T (inner) and K (outer)  -> val/test error + gap
#   Fig.2 analog (--mode fig2): SSGD (T = 1), vary K and m1   -> larger m1 helps
#   --mode lr       : learning-rate sensitivity (inner/outer step sizes)
#   --mode stability: empirical on-average argument stability (Definition 5)
#   --mode smoke    : fast end-to-end sanity check
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import csv
import argparse
import datetime
import logging

import torch
import numpy as np

# tiny batches (paper: 8) are much faster single-threaded; parallelism comes
# from running several experiment processes concurrently
torch.set_num_threads(int(os.environ.get('SBO_NUM_THREADS', '1')))

from core.ablo import train as ablo_train, measure_stability
from core.loss import ReweightingMLP
from core.datasets import CorruptedMnist
from core.utils import set_seed, set_logger, save_record_csv, average_records, plot_record

RECORD_KEYS = ["loss_tr_sgd", "loss_val_batch", "loss_val", "zero_one_loss_val",
               "loss_te", "zero_one_loss_te", "gap", "zero_one_gap"]


def default_config():
    return dict(
        dataset='corrupted_mnist', width=28, x_dim=784,
        theta_mlp_shape=[784, 256, 10],
        m_tr=2000, m_val=2000, m_te=1000,   # paper: 2000 / 2000 / 1000
        batch_size=8,                        # paper: initial batch size 8
        eval_batch_size=500,
        K=5000, T=32,                        # paper: K up to 5000, T = 32 in Fig.1 / T = 1 in Fig.2
        lr_l=0.01, lr_h=5.0,                 # paper: initial inner / outer step sizes
        wd_l=0., wd_h=0., mm_h=0.,
        eval_every=50,
    )


def build_data(cfg, m_mval=0):
    """Corrupted MNIST: 50% of the training labels replaced by random labels
    (paper Sec. 5.1); validation/test are clean and share the same distribution."""
    gen = CorruptedMnist(cfg['width'], 'classification', flatten=True)
    return gen.get_data(cfg['m_tr'], cfg['m_val'], cfg['m_te'], m_mval, cfg['x_dim'])


def run_once(cfg, seed, re_init=False, val_dataset=None, tr_dataset=None, te_dataset=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(seed)
    loss_cls = ReweightingMLP(cfg['m_tr'], cfg['theta_mlp_shape'])
    record = {k: [] for k in RECORD_KEYS}
    ablo_train(tr_dataset, val_dataset, te_dataset, loss_cls, cfg['K'], cfg['T'],
               cfg['lr_l'], cfg['wd_l'], cfg['lr_h'], cfg['wd_h'], cfg['mm_h'],
               cfg['batch_size'], cfg['batch_size'], cfg['eval_batch_size'], device, record,
               re_init=re_init, eval_every=cfg['eval_every'])
    return record


def summarize_and_save(configs, out_dir, tag):
    """Average curves over seeds for each config; save one CSV + plots."""
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []
    for name, records in configs:
        mean_rec, std_rec = average_records(records)
        plot_record(mean_rec, os.path.join(out_dir, name))
        for it in sorted(dict(mean_rec['gap']).keys()):
            row = {'config': name, 'iteration': it}
            for key in mean_rec:
                row[key] = dict(mean_rec[key])[it]
                row[key + '_std'] = dict(std_rec[key])[it]
            all_rows.append(row)
    path = os.path.join(out_dir, '{}_summary.csv'.format(tag))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    logging.info('saved {} ({} configs)'.format(path, len(configs)))
    return path


def final_stats(configs):
    lines = []
    for name, records in configs:
        mean_rec, _ = average_records(records)
        last_it = max(dict(mean_rec['gap']).keys())
        d = {k: dict(mean_rec[k])[last_it] for k in ['loss_val', 'loss_te', 'gap', 'zero_one_loss_val', 'zero_one_loss_te']}
        lines.append('{:>24s} @K={:5d}: val={:.4f} te={:.4f} gap={:+.4f} (0-1: val={:.4f} te={:.4f})'.format(
            name, last_it, d['loss_val'], d['loss_te'], d['gap'], d['zero_one_loss_val'], d['zero_one_loss_te']))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='smoke', choices=['smoke', 'fig1', 'fig2', 'lr', 'stability', 'all'])
    parser.add_argument('--K_max', type=int, default=5000, help='max outer iterations (paper: 5000)')
    parser.add_argument('--T_list', type=int, nargs='+', default=[1, 32], help='inner iterations per outer loop (Fig.1)')
    parser.add_argument('--m1_list', type=int, nargs='+', default=[500, 2000], help='validation set sizes (Fig.2)')
    parser.add_argument('--lr_l_list', type=float, nargs='+', default=[0.001, 0.01, 0.1])
    parser.add_argument('--lr_h_list', type=float, nargs='+', default=[0.5, 5.0])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 52, 62, 72, 82], help='paper: 5 repetitions')
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--out_root', default='workspace/exp')
    args = parser.parse_args()

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    out_dir = os.path.join(args.out_root, '{}_{}'.format(args.mode, now))
    os.makedirs(out_dir, exist_ok=True)
    set_logger(os.path.join(out_dir, 'log.txt'))
    set_seed(2024)  # fixed seed for the data-split stream (reproducible splits)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info('device: {}'.format(device))

    if args.mode == 'smoke':
        cfg = default_config()
        cfg.update(K=300, T=4, m_tr=400, m_val=500, m_te=500, eval_every=25)
        tr, val, te, _ = build_data(cfg)
        rec = run_once(cfg, seed=42, tr_dataset=tr, val_dataset=val, te_dataset=te)
        d = {k: dict(rec[k]) for k in ['loss_val', 'loss_te', 'gap']}
        it0, it1 = min(d['loss_val']), max(d['loss_val'])
        logging.info('smoke: loss_val {:.4f} -> {:.4f} | loss_te {:.4f} -> {:.4f} | gap {:+.4f} -> {:+.4f}'.format(
            d['loss_val'][it0], d['loss_val'][it1], d['loss_te'][it0], d['loss_te'][it1], d['gap'][it0], d['gap'][it1]))
        assert d['loss_val'][it1] < d['loss_val'][it0], 'validation loss did not decrease'
        assert all(np.isfinite(v) for v in d['gap'].values()), 'NaN in gap'
        save_record_csv(rec, os.path.join(out_dir, 'smoke_record.csv'))
        print('SMOKE OK: bilevel training converges, gap recorded, no NaN.')
        return

    if args.mode in ('fig1', 'all'):
        configs = []
        for T in args.T_list:
            cfg = default_config(); cfg.update(K=args.K_max, T=T, eval_every=args.eval_every)
            tr, val, te, _ = build_data(cfg)
            records = [run_once(cfg, seed, tr_dataset=tr, val_dataset=val, te_dataset=te) for seed in args.seeds]
            configs.append(('TSGD_T{}_K{}'.format(T, args.K_max), records))
        summarize_and_save(configs, out_dir, 'fig1')
        for line in final_stats(configs):
            logging.info(line)

    if args.mode in ('fig2', 'all'):
        configs = []
        for m1 in args.m1_list:
            cfg = default_config(); cfg.update(K=args.K_max, T=1, m_val=m1, eval_every=args.eval_every)
            tr, val, te, _ = build_data(cfg)
            records = [run_once(cfg, seed, tr_dataset=tr, val_dataset=val, te_dataset=te) for seed in args.seeds]
            configs.append(('SSGD_T1_m1{}_K{}'.format(m1, args.K_max), records))
        summarize_and_save(configs, out_dir, 'fig2')
        for line in final_stats(configs):
            logging.info(line)

    if args.mode in ('lr', 'all'):
        configs = []
        for lr_l in args.lr_l_list:
            for lr_h in args.lr_h_list:
                cfg = default_config(); cfg.update(K=args.K_max, T=1, lr_l=lr_l, lr_h=lr_h, eval_every=args.eval_every)
                tr, val, te, _ = build_data(cfg)
                records = [run_once(cfg, seed, tr_dataset=tr, val_dataset=val, te_dataset=te) for seed in args.seeds]
                configs.append(('SSGD_lrl{}_lrh{}'.format(lr_l, lr_h), records))
        summarize_and_save(configs, out_dir, 'lr')
        for line in final_stats(configs):
            logging.info(line)

    if args.mode in ('stability', 'all'):
        # Empirical l2 on-average argument stability (Definition 5): the drift
        # ||A(D) - A(D^(i))||_2 when one validation sample is replaced should
        # shrink with m1 (bounds in Table 1 scale like O(K/m1) for SSGD).
        n_pert = 3
        rows = []
        for m1 in args.m1_list:
            for K in sorted({min(args.K_max, 1000), min(args.K_max, 2000)}):
                cfg = default_config(); cfg.update(K=K, T=1, m_val=m1, eval_every=K + 1)
                tr, val, te, pool = build_data(cfg, m_mval=n_pert)  # pool: independent clean draws from D1
                perturb_indices = list(range(0, m1, max(1, m1 // n_pert)))[:n_pert]
                res = measure_stability(
                    tr, val, te, ReweightingMLP(cfg['m_tr'], cfg['theta_mlp_shape']),
                    K, cfg['T'], cfg['lr_l'], cfg['wd_l'], cfg['lr_h'], cfg['wd_h'], cfg['mm_h'],
                    cfg['batch_size'], cfg['batch_size'], cfg['eval_batch_size'], device,
                    perturb_indices, perturb_pool=list(pool.array), seed=args.seeds[0])
                for r in res:
                    rows.append({'m1': m1, 'K': K, **r})
                logging.info('stability m1={} K={}: mean dxy2={:.3e}'.format(
                    m1, K, res[-1]['dxy2']))
        path = os.path.join(out_dir, 'stability.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logging.info('saved {}'.format(path))


if __name__ == '__main__':
    main()
