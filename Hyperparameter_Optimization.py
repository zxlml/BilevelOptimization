# Notice that in this code, K and T stand for the numbers of inner and outer iterations.
from core.run import run
import datetime
import argparse
import random


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    seed=random.randint(1,1000)
    for KK in [2000]:
        args = argparse.Namespace(
            T=5000, K=KK, seed=seed, ho_algo='ablo', dataset='corrupted_mnist', width=28, x_dim=784,
            loss='ReweightingMLP', theta_mlp_shape=[784, 256, 10],
            m_tr=5000, m_val=2000, m_te=5000, m_mval=200, batch_size=32, lr_h=0.001, lr_l=0.001, wd_h=0., wd_l=0.,
            mm_h=0.)
        args.workspace_root = "workspace/runs/rw_mnist/Time_{}/Seed_{}_InnerLoop_{}".format(now, seed, KK)
        run(args)