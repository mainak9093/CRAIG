"""[REPLICATION PATCH 2026-06-12] Faithful PyTorch port of mnist.py (Fig. 4).

The original mnist.py uses Keras on TensorFlow 1.14, which cannot be installed on
modern Python. This port reproduces it exactly, preserving its behaviour:

  - data: MNIST train 60,000 / test 10,000, flattened to 784, kept at raw 0-255
    scale (the original never divides by 255 -- reproduced deliberately);
  - model: Dense(784->100) + sigmoid + Dense(100->10) + softmax, Glorot-uniform
    kernel init and zero bias (Keras defaults);
  - loss: categorical cross-entropy + l2(1e-4) kernel penalty on both Dense
    kernels, added to the loss exactly like keras.regularizers.l2 (penalty is
    included in reported train/test losses, as Keras evaluate() does);
  - optimizer: SGD, lr = 0.01 (Keras 'sgd' default), no momentum;
  - sample weights: per-batch Keras semantics  loss = sum(w*l) / sum(w);
  - loop: batch_size 32, epochs 15, runs 5; trains on the previous subset, then
    re-selects via CRAIG (features = softmax(logits) - onehot, per-class
    facility location through util.get_orders_and_weights, smtk=0) or uniformly
    at random; the first epoch of each run trains on the full set;
    fixed sequential batch slicing, no shuffling (as in the original);
  - results: identical .npz schema, saved to results/mnist_{size}_{grd}_{runs}.npz.

Differences from the original (all documented in REPLICATION.md):
  - flags are CLI arguments instead of editing code (defaults = original values);
  - `grd` is defined for the full-data case too (the original raised NameError
    when subset=False);
  - runs on CUDA when available (selection features are computed on GPU).

Run (office GPU):
  python mnist_torch.py                       # CRAIG, subset_size 0.4 (original flags)
  python mnist_torch.py --random              # random subsets baseline
  python mnist_torch.py --full                # full-data baseline
  python mnist_torch.py --subset_size 0.5     # the paper's Fig. 4 uses 50%
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

import util

p = argparse.ArgumentParser(description='CRAIG MNIST MLP (port of mnist.py)')
p.add_argument('--subset_size', '-s', type=float, default=0.4)
p.add_argument('--random', action='store_true', help='random subsets instead of CRAIG')
p.add_argument('--full', action='store_true', help='train on all data (no subsets)')
p.add_argument('--epochs', type=int, default=15)
p.add_argument('--runs', type=int, default=5)
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--reg', type=float, default=1e-4)
p.add_argument('--lr', type=float, default=0.01)
p.add_argument('--save_subset', action='store_true')
args = p.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPO = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(REPO, 'results'), exist_ok=True)
folder = os.path.join(REPO, 'results', 'mnist')

# ---- data (raw 0-255 floats, as in the original) ----
tr = datasets.MNIST(os.path.join(REPO, 'data'), train=True, download=True)
te = datasets.MNIST(os.path.join(REPO, 'data'), train=False, download=True)
X_train = tr.data.reshape(60000, 784).float()
Y_train_nocat = tr.targets.numpy()
Y_train = np.eye(10)[Y_train_nocat]
X_test = te.data.reshape(10000, 784).float()
Y_test = te.targets.numpy()

num_classes, smtk = 10, 0
subset = not args.full
rand = args.random
subset_size = args.subset_size if subset else 1.0
batch_size, epochs, runs, reg = args.batch_size, args.epochs, args.runs, args.reg

X_train_d = X_train.to(device)
Y_train_t = torch.from_numpy(Y_train_nocat).long().to(device)
X_test_d = X_test.to(device)
Y_test_t = torch.from_numpy(Y_test).long().to(device)


def make_model():
    m = nn.Sequential(nn.Linear(784, 100), nn.Sigmoid(), nn.Linear(100, 10))
    for layer in (m[0], m[2]):  # Keras Dense defaults: Glorot-uniform kernel, zero bias
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    return m.to(device)


def l2_penalty(m):
    return reg * (m[0].weight.pow(2).sum() + m[2].weight.pow(2).sum())


@torch.no_grad()
def evaluate(m, X, Y_t):
    """Keras model.evaluate: CE + regularization penalty, accuracy."""
    m.eval()
    losses, correct, n = 0.0, 0, len(X)
    pen = float(l2_penalty(m))
    for s in range(0, n, 4096):
        logits = m(X[s:s + 4096])
        losses += float(F.cross_entropy(logits, Y_t[s:s + 4096], reduction='sum'))
        correct += int((logits.argmax(1) == Y_t[s:s + 4096]).sum())
    return losses / n + pen, correct / n


@torch.no_grad()
def softmax_probs(m, X):
    m.eval()
    out = torch.zeros(len(X), num_classes, device=device)
    for s in range(0, len(X), 4096):
        out[s:s + 4096] = F.softmax(m(X[s:s + 4096]), dim=1)
    return out.cpu().numpy()


train_loss, test_loss = np.zeros((runs, epochs)), np.zeros((runs, epochs))
train_acc, test_acc = np.zeros((runs, epochs)), np.zeros((runs, epochs))
train_time = np.zeros((runs, epochs))
grd_time, sim_time, pred_time = np.zeros((runs, epochs)), np.zeros((runs, epochs)), np.zeros((runs, epochs))
not_selected = np.zeros((runs, epochs))
times_selected = np.zeros((runs, len(X_train)))
best_acc = 0
print(f'----------- smtk: {smtk}, device: {device} ------------')

if args.save_subset:
    B = int(subset_size * len(X_train))
    selected_ndx = np.zeros((runs, epochs, B))
    selected_wgt = np.zeros((runs, epochs, B))

grd = 'all'  # [PATCH] original left grd undefined when subset=False
for run in range(runs):
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    sub_idx = np.arange(len(X_train))            # first epoch trains on everything
    W_subset = np.ones(len(sub_idx))
    ordering_time, similarity_time, pre_time = 0, 0, 0
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        Xs = X_train_d[sub_idx]
        Ys = Y_train_t[sub_idx]
        Ws = torch.from_numpy(W_subset).float().to(device)
        num_batches = int(np.ceil(len(Xs) / float(batch_size)))

        model.train()
        for index in range(num_batches):       # fixed slicing, no shuffle (original)
            sl = slice(index * batch_size, (index + 1) * batch_size)
            xb, yb, wb = Xs[sl], Ys[sl], Ws[sl]
            start = time.time()
            ce = F.cross_entropy(model(xb), yb, reduction='none')
            loss = (wb * ce).sum() / wb.sum() + l2_penalty(model)  # Keras sample_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_time[run][epoch] += time.time() - start

        if subset:
            if rand:
                indices = np.arange(0, len(X_train))
                np.random.shuffle(indices)
                indices = indices[:int(subset_size * len(X_train))]
                W_subset = np.ones(len(indices))
            else:
                start = time.time()
                _logits = softmax_probs(model, X_train_d)
                pre_time = time.time() - start
                features = _logits - Y_train

                indices, W_subset, _, _, ordering_time, similarity_time = util.get_orders_and_weights(
                    int(subset_size * len(X_train)), features, 'euclidean', smtk, 0, False, Y_train_nocat)
                W_subset = W_subset / np.sum(W_subset) * len(W_subset)

            if args.save_subset:
                selected_ndx[run, epoch], selected_wgt[run, epoch] = indices, W_subset

            grd_time[run, epoch], sim_time[run, epoch], pred_time[run, epoch] = \
                ordering_time, similarity_time, pre_time
            times_selected[run][indices] += 1
            not_selected[run, epoch] = np.sum(times_selected[run] == 0) / len(times_selected[run]) * 100
            sub_idx = np.asarray(indices, dtype=np.int64)
        else:
            sub_idx = np.arange(len(X_train))

        test_loss[run][epoch], test_acc[run][epoch] = evaluate(model, X_test_d, Y_test_t)
        train_loss[run][epoch], train_acc[run][epoch] = evaluate(model, X_train_d, Y_train_t)
        best_acc = max(test_acc[run][epoch], best_acc)

        grd = 'random_wor' if rand else ('grd_normw' if subset else 'all')
        print(f'run: {run}, {grd}, subset_size: {subset_size}, epoch: {epoch}, '
              f'test_acc: {test_acc[run][epoch]:.4f}, loss: {train_loss[run][epoch]:.4f}, '
              f'best_prec1_gb: {best_acc:.4f}, not selected %:{not_selected[run][epoch]:.2f}')

    out = f'{folder}_{subset_size}_{grd}_{runs}'
    print(f'Saving the results to {out}')
    payload = dict(train_loss=train_loss, test_acc=test_acc, train_acc=train_acc,
                   test_loss=test_loss, train_time=train_time, grd_time=grd_time,
                   sim_time=sim_time, pred_time=pred_time,
                   not_selected=not_selected, times_selected=times_selected)
    if args.save_subset:
        payload.update(subset=selected_ndx, weights=selected_wgt)
    np.savez(out, **payload)
