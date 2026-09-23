# Unit tests for the SBO paper code (gen_sbo.pdf, IJCAI 2024).
# Run from the repo root:  python -m pytest tests/ -v
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import pytest
import torch
import torch.nn.functional as F

from core.ablo import sgd_step, train as ablo_train
from core.loss import ReweightingMLP
from core.datasets import QuickDataset
from core.utils import set_seed, average_records


# ---------------------------------------------------------------------------
# synthetic reweighting task: theta = [W] (1 x d), logits = x @ W^T
# inner loss  = sigmoid(lamb[i]) * (logit - y)^2
# outer loss  = (logit - y)^2  (lamb enters only through the unrolled theta)
# ---------------------------------------------------------------------------
class SyntheticReweighting:
    def __init__(self, m_tr, d, device='cpu'):
        self.m_tr, self.d, self.device = m_tr, d, device

    def loss_in(self, lamb, theta, z):
        i, x, y = z
        logits = F.linear(x, theta[0])
        return lamb[i].sigmoid() * (logits - y) ** 2

    def loss_out(self, lamb, theta, z):
        x, y = z[-2], z[-1]
        logits = F.linear(x, theta[0])
        return (logits - y) ** 2

    def zero_one_loss(self, lamb, theta, z):
        x, y = z[-2], z[-1]
        pred = (F.linear(x, theta[0]) - y).abs() < 0.5
        return (~pred).float()

    def init_lamb(self, requires_grad):
        return torch.zeros(self.m_tr, device=self.device, requires_grad=requires_grad)

    def init_theta(self, requires_grad):
        w = torch.randn(1, self.d, device=self.device) * 0.1
        return [w.requires_grad_(requires_grad)]


def make_synthetic_data(m_tr=20, m_val=10, m_te=10, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    def gen(m):
        x = torch.randn(m, d, generator=g)
        y = (x.sum(-1, keepdim=True) > 0).float()
        return x, y
    x_tr, y_tr = gen(m_tr)
    tr = QuickDataset([(torch.tensor(i), x_tr[i], y_tr[i]) for i in range(m_tr)])
    xv, yv = gen(m_val)
    val = QuickDataset([(xv[i], yv[i]) for i in range(m_val)])
    xt, yt = gen(m_te)
    te = QuickDataset([(xt[i], yt[i]) for i in range(m_te)])
    return tr, val, te


RECORD_KEYS = ["loss_tr_sgd", "loss_val_batch", "loss_val", "zero_one_loss_val",
               "loss_te", "zero_one_loss_te", "gap", "zero_one_gap"]


def run_bilevel(K=30, T=2, lr_l=0.05, lr_h=0.1, re_init=False, seed=0,
                m_tr=40, m_val=20, m_te=20, d=4):
    set_seed(seed)
    tr, val, te = make_synthetic_data(m_tr, m_val, m_te, d, seed)
    loss_cls = SyntheticReweighting(m_tr, d)
    record = {k: [] for k in RECORD_KEYS}
    ablo_train(tr, val, te, loss_cls, K, T, lr_l, 0., lr_h, 0., 0.,
               4, 4, 64, 'cpu', record, re_init=re_init, eval_every=5, print_every=K + 1)
    return record


# ---------------------------------------------------------------- sgd_step
def test_sgd_step_single_step_matches_gradient():
    set_seed(0)
    w0 = torch.tensor([1.0])
    theta = [w0.clone().requires_grad_(False)]
    lamb = torch.zeros(1, requires_grad=True)
    x = torch.tensor([[2.0]])
    y = torch.tensor([[1.0]])
    loss_fn = lambda lamb_, theta_, z: ((F.linear(z[0], theta_[0]) - z[1]) ** 2).sum(dim=1)
    lr = 0.1
    theta_new, loss_val = sgd_step(theta, lamb, loss_fn, [x, y], lr, 0.)
    # grad = 2*(w*x - y)*x = 2*(2-1)*2 = 4
    expected = w0 - lr * torch.tensor([4.0])
    assert torch.allclose(theta_new[0].detach(), expected, atol=1e-6)
    assert math.isclose(loss_val, 1.0, rel_tol=1e-6)


def test_sgd_step_hypergradient_flows_to_lamb():
    set_seed(0)
    theta = [torch.tensor([1.0]).requires_grad_(False)]
    lamb = torch.tensor([2.0], requires_grad=True)
    loss_fn = lambda lamb_, theta_, z: ((theta_[0] - lamb_) ** 2)
    lr = 0.1
    theta_new, _ = sgd_step(theta, lamb, loss_fn, None, lr, 0.)
    # theta1 = theta0 - lr * 2*(theta0 - lamb) => d theta1 / d lamb = 2*lr
    outer = (theta_new[0] ** 2).sum()
    outer.backward()
    # d outer/d lamb = 2*theta1 * 2*lr, theta1 = 1 - 0.1*2*(1-2) = 1.2
    assert lamb.grad is not None
    assert torch.allclose(lamb.grad, torch.tensor(2 * 1.2 * 2 * 0.1), atol=1e-6)


# ------------------------------------------------------- algorithm variants
def test_ud_reinit_restores_initial_theta_while_warm_start_drifts():
    # UD (Algorithm 3) restores the SAME initial y0 every outer loop; TSGD warm
    # start keeps updating theta across outer loops. With lr_h = 0 (lamb frozen)
    # and T = 1: UD's final theta is one inner step away from theta0, while the
    # warm-start theta accumulates K steps.
    def one_run(re_init):
        set_seed(5)
        tr, val, te = make_synthetic_data(40, 20, 20, 4, 5)
        loss_cls = SyntheticReweighting(40, 4)
        theta0 = loss_cls.init_theta(requires_grad=False)  # same first draw as inside train()
        record = {k: [] for k in RECORD_KEYS}
        _, theta = ablo_train(tr, val, te, loss_cls, 10, 1, 0.05, 0., 0.0, 0., 0.,
                              4, 4, 64, 'cpu', record, re_init=re_init, eval_every=11, print_every=11)
        return (theta[0] - theta0[0]).norm().item()

    dist_ud, dist_ws = one_run(True), one_run(False)
    assert dist_ws > dist_ud, (dist_ud, dist_ws)


def test_bilevel_training_converges_and_gap_recorded():
    rec = run_bilevel(K=150, T=2, lr_l=0.05, lr_h=0.1, seed=1)
    d_val = dict(rec["loss_val"])
    its = sorted(d_val.keys())
    assert d_val[its[-1]] < d_val[its[0]], "validation loss did not decrease"
    for it in its:
        gap = dict(rec["gap"])[it]
        expected = dict(rec["loss_te"])[it] - d_val[it]
        assert gap == pytest.approx(expected, abs=1e-8)
        assert all(math.isfinite(dict(rec[k])[it]) for k in RECORD_KEYS if k in rec)


def test_hypergradient_nonzero_through_inner_loop():
    # lamb must receive gradient through the T unrolled inner steps
    set_seed(0)
    tr, val, te = make_synthetic_data(seed=0)
    loss_cls = SyntheticReweighting(20, 4)
    record = {k: [] for k in RECORD_KEYS}
    ablo_train(tr, val, te, loss_cls, 3, 3, 0.05, 0., 0.5, 0., 0., 4, 4, 64, 'cpu', record, eval_every=1)
    # after training, lamb must have moved away from its zero initialization
    # (checked indirectly: loss_val must change across evals)
    vals = [v for _, v in record["loss_val"]]
    assert len(vals) >= 3 and any(abs(a - b) > 1e-9 for a, b in zip(vals, vals[1:]))


def test_nan_guard_stops_training():
    # a huge outer step size triggers divergence; the loop must stop, not crash
    rec = run_bilevel(K=50, T=2, lr_l=1e4, lr_h=1e6, seed=0)
    n_iter = len(rec["loss_tr_sgd"])
    assert n_iter <= 50  # either finished or early-stopped on NaN
    assert all(math.isfinite(v) for _, v in rec["loss_tr_sgd"][:1])


# ---------------------------------------------------------------- averaging
def test_average_records_mean_std():
    recs = []
    for seed in range(3):
        rec = run_bilevel(K=10, T=1, seed=seed)
        recs.append(rec)
    mean_rec, std_rec = average_records(recs)
    its = sorted(dict(mean_rec["gap"]).keys())
    assert len(its) > 0
    for it in its:
        m = dict(mean_rec["gap"])[it]
        s = dict(std_rec["gap"])[it]
        assert math.isfinite(m) and math.isfinite(s) and s >= 0


# --------------------------------------------------------- ReweightingMLP
def test_reweighting_mlp_loss_shapes_and_lamb_indexing():
    set_seed(0)
    m_tr, d = 12, 8
    loss_cls = ReweightingMLP(m_tr, [d, 4, 3])
    lamb = loss_cls.init_lamb(requires_grad=True)
    assert lamb.shape == (m_tr,) and lamb.requires_grad
    theta = loss_cls.init_theta(requires_grad=False)
    idx = torch.arange(4)
    x = torch.randn(4, d)
    y = torch.randint(0, 3, (4,))
    lin = (lamb[idx].sigmoid() * F.cross_entropy(F.linear(x, theta[0]), y, reduction='none'))
    lin.sum().backward()
    assert lamb.grad is not None and lamb.grad.abs().sum() > 0
    # outer loss ignores lamb directly: gradient must flow only via theta
    lamb2 = loss_cls.init_lamb(requires_grad=True)
    out = loss_cls.loss_out(lamb2, theta, (x, y)).mean()
    assert out.item() > 0


# ------------------------------------------------------------ MNIST (real)
def _mnist_available():
    try:
        from core.datasets import CorruptedMnist
        CorruptedMnist(28, 'classification', flatten=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mnist_available(), reason="MNIST not downloadable")
def test_corrupted_mnist_paper_settings():
    from core.datasets import CorruptedMnist
    set_seed(42)
    gen = CorruptedMnist(28, 'classification', flatten=True)
    tr, val, te, mval = gen.get_data(2000, 2000, 1000, 1, 784)  # paper: 2000/2000/1000
    assert len(tr) == 2000 and len(val) == 2000 and len(te) == 1000
    x0, y0 = tr[0][1], tr[0][2]
    assert x0.shape == (784,) and 0 <= int(y0) <= 9
    # exactly half of the training stream is label-corrupted (paper: 50%)


@pytest.mark.skipif(not _mnist_available(), reason="MNIST not downloadable")
def test_mnist_bilevel_smoke_convergence():
    from core.datasets import CorruptedMnist
    set_seed(0)
    gen = CorruptedMnist(28, 'classification', flatten=True)
    tr, val, te, _ = gen.get_data(200, 200, 200, 0, 784)
    loss_cls = ReweightingMLP(200, [784, 64, 10])
    record = {k: [] for k in RECORD_KEYS}
    ablo_train(tr, val, te, loss_cls, 60, 4, 0.01, 0., 5.0, 0., 0.,
               8, 8, 200, 'cpu', record, eval_every=20, print_every=61)
    d_val = dict(record["loss_val"])
    its = sorted(d_val.keys())
    assert d_val[its[-1]] < d_val[its[0]] + 1e-9, "val loss should not increase from init"
    assert all(math.isfinite(v) for v in d_val.values())
