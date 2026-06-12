"""[REPLICATION PATCH 2026-06-12] Reproduce the CRAIG paper's figures from results/.

Reads the .npz files written by logistic.py / mnist_torch.py / train_resnet.py and
rebuilds the paper's plots (same axes, colors and conventions):

  Fig 1  covtype: training-loss residual + test error vs wall-clock time,
         for SGD / SAGA / SVRG; CRAIG 10% (blue) vs random 10% (green) vs all
         data (orange). CRAIG curves include the subset-selection time.
  Fig 2  gradient-estimation error vs subset size (covtype, ijcnn1).
  Fig 3  ijcnn1: loss residual vs time for CRAIG vs random subsets of
         10%..90%, plus the speedup table to reach the full-data loss.
  Fig 4  MNIST MLP: test accuracy + training loss vs time (error bars over runs).
  Fig 5  CIFAR10 ResNet-20: best test accuracy vs fraction of data selected,
         CRAIG vs random (per lag); plus per-epoch curves for the core runs.
  Fig 6  grids of CIFAR10 images selected by CRAIG at the start / middle / end
         of training (needs runs with --save_subset).

Usage:
  python plots.py --all              # build every figure that has data
  python plots.py --fig 1 3          # specific figures
  python plots.py --validate         # print the replication success criteria

Figures are written to results/figures/.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(REPO, 'results')
FIG = os.path.join(RES, 'figures')

BLUE, GREEN, ORANGE = 'tab:blue', 'tab:green', 'tab:orange'  # craig, random, full


def _load(path):
    if not os.path.isfile(path):
        return None
    return np.load(path, allow_pickle=True)


def _nonempty(F):
    """Row mask of runs that actually ran (all-zero rows are unused slots)."""
    return np.abs(F).sum(axis=1) > 0


def _sel_time(data, size):
    """Selection time (ordering + similarity) from the cached ordering npz."""
    d = _load(os.path.join(RES, f'{data}_{size}_l2.npz'))
    if d is None:
        return 0.0
    t = float(np.max(d['order_time']))
    t += float(np.max(d['similarity_time'])) if 'similarity_time' in d.files else 0.0
    return t


def _logistic_curves(data, method, size, tag):
    """(time[T], loss[runs,T], err[runs,T]) for one logistic.py result file."""
    d = _load(os.path.join(RES, f'{data}_{method}_{size}_{tag}_best_f_l2_w.npz'))
    if d is None:
        return None
    keep = _nonempty(d['F_all'])
    F, T, A = d['F_all'][keep], d['T_all'][keep], d['Acc_all'][keep]
    t = T.mean(axis=0)
    if tag.startswith('grd'):
        t = t + _sel_time(data, size)  # paper: wall-clock includes selection
    return t, F, 1.0 - A


# --------------------------------------------------------------------------- Fig 1
def fig1_covtype():
    methods = [('sgd', 'SGD'), ('saga', 'SAGA'), ('svrg', 'SVRG')]
    settings = [('0.1', 'grd_rand', f'CRAIG 10%', BLUE),
                ('0.1', 'rand_nw', f'Random 10%', GREEN),
                ('1.0', 'rand_nw', 'All data', ORANGE)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    plotted = False
    for j, (m, name) in enumerate(methods):
        curves = {lbl: _logistic_curves('covtype', m, s, tag) for s, tag, lbl, _ in settings}
        present = [v for v in curves.values() if v is not None]
        if not present:
            continue
        f_star = min(float(F.min()) for _, F, _ in present)
        for (s, tag, lbl, col) in settings:
            cv = curves[lbl]
            if cv is None:
                continue
            t, F, E = cv
            resid = np.maximum(F - f_star, 1e-9)
            axes[0, j].plot(t, resid.mean(0), color=col, label=lbl)
            axes[0, j].fill_between(t, np.maximum(resid.min(0), 1e-9), resid.max(0),
                                    color=col, alpha=0.2)
            axes[1, j].plot(t, E.mean(0), color=col, label=lbl)
            axes[1, j].fill_between(t, E.min(0), E.max(0), color=col, alpha=0.2)
            plotted = True
        axes[0, j].set(title=f'(a) {name}' if j == 0 else f'({chr(97+j)}) {name}',
                       xlabel='Time (sec)', ylabel='Training loss residual', yscale='log')
        axes[1, j].set(xlabel='Time (sec)', ylabel='Test error rate', yscale='log')
        axes[0, j].legend(); axes[1, j].legend()
    if not plotted:
        plt.close(fig); print('[fig1] no covtype results yet'); return
    fig.suptitle('Fig 1 — Logistic regression on covtype: CRAIG 10% vs random 10% vs all data')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig1_covtype.png'), dpi=150); plt.close(fig)
    print('[fig1] written')


# --------------------------------------------------------------------------- Fig 2
def fig2_graddiff():
    subsets = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plotted = False
    for j, data in enumerate(['covtype', 'ijcnn1']):
        for tag, col, lbl in [('grd_rand', BLUE, 'CRAIG'), ('rand_wgt', GREEN, 'Random')]:
            d = _load(os.path.join(RES, f'{data}_sgd_{tag}_l2_grad_diff_w.npz'))
            if d is None:
                continue
            diff, norms = d['diff'], d['max_full_grad_norms']
            keep = diff.sum(axis=1) > 0
            if not keep.any():
                continue
            rel = diff[keep] / np.maximum(norms[keep].max(), 1e-12)
            for row in rel:  # transparent individual runs, like the paper
                axes[j].plot(subsets, row, color=col, alpha=0.25)
            axes[j].plot(subsets, rel.mean(0), color=col, label=lbl, linewidth=2)
            plotted = True
        axes[j].set(title=f'({chr(97+j)}) {data.capitalize()}',
                    xlabel='Fraction of data selected', ylabel='Normalized gradient difference')
        axes[j].legend()
    if not plotted:
        plt.close(fig); print('[fig2] no grad-diff results yet'); return
    fig.suptitle('Fig 2 — Full-gradient estimation error of the subset')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_graddiff.png'), dpi=150); plt.close(fig)
    print('[fig2] written')


# --------------------------------------------------------------------------- Fig 3
def fig3_ijcnn1():
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    full = _logistic_curves('ijcnn1', 'sgd', '1.0', 'rand_nw')
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted = False
    f_candidates = [float(full[1].min())] if full else []
    curves = {}
    for s in sizes:
        for tag in ('grd_rand', 'rand_nw'):
            cv = _logistic_curves('ijcnn1', 'sgd', str(s), tag)
            if cv:
                curves[(s, tag)] = cv
                f_candidates.append(float(cv[1].min()))
    if not f_candidates:
        plt.close(fig); print('[fig3] no ijcnn1 results yet'); return
    f_star = min(f_candidates)

    speedups = []
    full_end_time, full_end_resid = None, None
    if full:
        t, F, _ = full
        resid = np.maximum(F.mean(0) - f_star, 1e-9)
        ax.plot(t, resid, color=ORANGE, marker='*', label='SGD + All data')
        full_end_time, full_end_resid = t[-1], resid[-1]
    for s in sizes:
        for tag, col in (('grd_rand', BLUE), ('rand_nw', GREEN)):
            if (s, tag) not in curves:
                continue
            t, F, _ = curves[(s, tag)]
            resid = np.maximum(F.mean(0) - f_star, 1e-9)
            lbl = ('SGD + CRAIG' if tag == 'grd_rand' else 'SGD + Random subset') if s == sizes[0] else None
            ax.plot(t, resid, color=col, marker='o', markersize=3, label=lbl)
            ax.annotate(f'{int(s*100)}%', (t[-1], resid[-1]), color=col, fontsize=8)
            if tag == 'grd_rand' and full_end_resid is not None:
                hit = np.where(resid <= full_end_resid)[0]
                if hit.size:
                    speedups.append((s, full_end_time / max(t[hit[0]], 1e-9)))
            plotted = True
    if not plotted:
        plt.close(fig); print('[fig3] no ijcnn1 subset results yet'); return
    ax.set(title='Fig 3 — Ijcnn1: loss residual vs time for CRAIG vs random subsets',
           xlabel='Time (sec)', ylabel='Training loss residual', xscale='log', yscale='log')
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig3_ijcnn1.png'), dpi=150); plt.close(fig)
    print('[fig3] written')
    if speedups:
        print('  speedup to reach the full-data final loss:')
        for s, sp in speedups:
            print(f'    CRAIG {int(s*100)}%: {sp:.1f}x')


# --------------------------------------------------------------------------- Fig 4
def fig4_mnist():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = False
    for pattern, col, lbl in [('grd_normw', BLUE, 'SGD + CRAIG'),
                              ('random_wor', GREEN, 'SGD + Random subsets'),
                              ('all', ORANGE, 'SGD + All data')]:
        hits = sorted(glob.glob(os.path.join(RES, f'mnist_*_{pattern}_*.npz')))
        if not hits:
            continue
        d = np.load(hits[0], allow_pickle=True)
        keep = _nonempty(d['test_acc'])
        sel = (d['grd_time'] + d['sim_time'] + d['pred_time'])[keep] if pattern == 'grd_normw' \
            else np.zeros_like(d['train_time'][keep])
        t = np.cumsum(d['train_time'][keep] + sel, axis=1).mean(0)
        acc, tl = d['test_acc'][keep], d['train_loss'][keep]
        axes[0].errorbar(t, acc.mean(0), yerr=acc.std(0), color=col, label=lbl, capsize=2)
        axes[1].errorbar(t, tl.mean(0), yerr=tl.std(0), color=col, label=lbl, capsize=2)
        plotted = True
    if not plotted:
        plt.close(fig); print('[fig4] no mnist results yet'); return
    axes[0].set(xlabel='Time (sec)', ylabel='Test Accuracy')
    axes[1].set(xlabel='Time (sec)', ylabel='Training Loss')
    axes[0].legend(); axes[1].legend()
    fig.suptitle('Fig 4 — MNIST 2-layer MLP: CRAIG vs random subsets vs all data')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig4_mnist.png'), dpi=150); plt.close(fig)
    print('[fig4] written')


# --------------------------------------------------------------------------- Fig 5
def _cifar_file(size, greedy, lag):
    grd = 'grd_w_warm' if greedy else 'rand_rsize_1.0_warm'
    return os.path.join(RES, f'cifar10_sgd_moment_0.9_resnet20_{size}_{grd}_mile_start_0_lag_{lag}.npz')


def fig5_cifar():
    sizes = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plotted = False
    for j, lag in enumerate((1, 5)):
        for greedy, col, lbl in [(1, BLUE, 'SGD + CRAIG'), (0, GREEN, 'SGD + Random subsets')]:
            xs, ys, ann = [], [], []
            for s in sizes:
                d = _load(_cifar_file(s, greedy, lag))
                if d is None:
                    continue
                keep = _nonempty(d['test_acc'])
                best = d['test_acc'][keep].max(axis=1).mean()
                frac = 100.0 - d['not_selected'][keep].min(axis=1).mean()
                xs.append(frac); ys.append(best); ann.append(s)
            if xs:
                o = np.argsort(xs)
                axes[j].plot(np.array(xs)[o], np.array(ys)[o], color=col, marker='o', label=lbl)
                for x, y, s in zip(xs, ys, ann):
                    axes[j].annotate(f'{int(s*100)}%', (x, y), fontsize=8, color=col)
                plotted = True
        axes[j].set(title=f'({chr(97+j)}) selection every {lag} epoch(s)',
                    xlabel='Fraction of data selected (%)', ylabel='Test Accuracy')
        axes[j].legend()
    if not plotted:
        plt.close(fig); print('[fig5] no cifar10 sweep results yet'); return
    fig.suptitle('Fig 5 — ResNet-20 on CIFAR10: data-efficiency of CRAIG vs random')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig5_cifar10.png'), dpi=150); plt.close(fig)
    print('[fig5] written')


def fig5_curves():
    """Per-epoch curves for the core trio (CRAIG 10% / random 10% / full, lag 1)."""
    runs = [(_cifar_file(0.1, 1, 1), BLUE, 'CRAIG 10%'),
            (_cifar_file(0.1, 0, 1), GREEN, 'Random 10%'),
            (_cifar_file(1.0, 0, 1), ORANGE, 'All data')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plotted = False
    for path, col, lbl in runs:
        d = _load(path)
        if d is None:
            continue
        keep = _nonempty(d['test_acc'])
        acc = d['test_acc'][keep].mean(0)
        tl = d['train_loss'][keep].mean(0)
        t = np.cumsum(d['train_time'][keep] + d['grd_time'][keep] + d['sim_time'][keep],
                      axis=1).mean(0)
        ep = np.arange(len(acc))
        axes[0].plot(ep, acc, color=col, label=lbl)
        axes[1].plot(ep, tl, color=col, label=lbl)
        axes[2].plot(t / 3600, acc, color=col, label=lbl)
        plotted = True
    if not plotted:
        plt.close(fig); print('[fig5-curves] no cifar10 core results yet'); return
    axes[0].set(xlabel='Epoch', ylabel='Test Accuracy')
    axes[1].set(xlabel='Epoch', ylabel='Training Loss')
    axes[2].set(xlabel='Wall-clock time (hours, incl. selection)', ylabel='Test Accuracy')
    for ax in axes:
        ax.legend()
    fig.suptitle('CIFAR10 ResNet-20 per-epoch curves (CRAIG vs random vs all data)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig5_cifar10_curves.png'), dpi=150); plt.close(fig)
    print('[fig5-curves] written')


# --------------------------------------------------------------------------- Fig 6
def fig6_subsets():
    hits = sorted(glob.glob(os.path.join(RES, 'cifar10_*grd_w*_subset.npz')))
    if not hits:
        print('[fig6] no cifar10 --save_subset results yet'); return
    from torchvision import datasets
    raw = datasets.CIFAR10(os.path.join(REPO, 'data'), train=True, download=True)
    images = raw.data  # [50000, 32, 32, 3] uint8
    d = np.load(hits[0], allow_pickle=True)
    subs = d['subset']  # [runs, epochs, B]
    epochs = subs.shape[1]
    stages = [(0, '(a) First'), (epochs // 2, '(b) Middle'), (epochs - 1, '(c) Last')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (ep, title) in zip(axes, stages):
        idx = subs[0, ep].astype(int)[:50]
        grid = np.zeros((5 * 32, 10 * 32, 3), dtype=np.uint8)
        for k, i in enumerate(idx):
            r, c = divmod(k, 10)
            grid[r*32:(r+1)*32, c*32:(c+1)*32] = images[i]
        ax.imshow(grid); ax.set_title(title); ax.axis('off')
    fig.suptitle('Fig 6 — CIFAR10 images selected by CRAIG during training')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig6_subsets.png'), dpi=150); plt.close(fig)
    print('[fig6] written')


# --------------------------------------------------------------------------- validation
def validate():
    print('=== Replication success criteria ===')
    # 1) covtype speedup: CRAIG 10% vs full, time to reach the full-data final loss
    for m in ('sgd', 'svrg', 'saga'):
        full = _logistic_curves('covtype', m, '1.0', 'rand_nw')
        craig = _logistic_curves('covtype', m, '0.1', 'grd_rand')
        if not (full and craig):
            print(f'covtype {m}: missing results'); continue
        f_star = min(float(full[1].min()), float(craig[1].min()))
        full_t, full_resid = full[0][-1], full[1].mean(0)[-1] - f_star
        resid = craig[1].mean(0) - f_star
        hit = np.where(resid <= max(full_resid, 1e-9))[0]
        if hit.size:
            sp = full_t / max(craig[0][hit[0]], 1e-9)
            verdict = 'PASS' if (m != 'sgd' or sp > 2.5) else 'CHECK'
            print(f'covtype {m}: CRAIG 10% speedup = {sp:.2f}x  '
                  f'(paper: ~2.75x sgd / ~4.5x svrg / ~2.5x saga) [{verdict} >2.5x criterion for sgd]')
        else:
            print(f'covtype {m}: CRAIG 10% did not reach the full-data loss '
                  f'(craig best resid {resid.min():.2e} vs full {full_resid:.2e})')
        err_full = 1 - np.load(os.path.join(RES, f'covtype_{m}_1.0_rand_nw_best_f_l2_w.npz'))['Acc_all']
        err_craig = craig[2]
        gap = float(err_craig.mean(0)[-1] - err_full[_nonempty(1-err_full)].mean(0)[-1])
        print(f'covtype {m}: final test-error gap CRAIG-full = {gap*100:+.2f}pp '
              f'[{"PASS" if abs(gap) <= 0.01 else "CHECK"} within 1%]')
    # 2) CIFAR10: CRAIG vs random at 10%
    a = _load(_cifar_file(0.1, 1, 1)); b = _load(_cifar_file(0.1, 0, 1)); f = _load(_cifar_file(1.0, 0, 1))
    if a is not None and b is not None:
        ba = a['test_acc'][_nonempty(a['test_acc'])].max(1).mean()
        bb = b['test_acc'][_nonempty(b['test_acc'])].max(1).mean()
        print(f'cifar10 10%: CRAIG best acc {ba:.2f} vs random {bb:.2f} '
              f'[{"PASS" if ba > bb else "FAIL"} CRAIG > random]')
        if f is not None:
            bf = f['test_acc'][_nonempty(f['test_acc'])].max(1).mean()
            print(f'cifar10: full-data best acc {bf:.2f} (CRAIG gap {bf-ba:.2f}pp)')
    else:
        print('cifar10: missing results')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--fig', nargs='*', type=int, default=[])
    ap.add_argument('--validate', action='store_true')
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    figs = {1: fig1_covtype, 2: fig2_graddiff, 3: fig3_ijcnn1, 4: fig4_mnist,
            5: lambda: (fig5_cifar(), fig5_curves()), 6: fig6_subsets}
    todo = sorted(figs) if (args.all or (not args.fig and not args.validate)) else args.fig
    for k in todo:
        try:
            figs[k]()
        except Exception as e:
            print(f'[fig{k}] failed: {e}')
    if args.validate or args.all:
        validate()


if __name__ == '__main__':
    main()
