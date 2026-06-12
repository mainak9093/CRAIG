"""[REPLICATION PATCH 2026-06-12] One-command driver for the CRAIG replication.

Runs the paper's experiments stage by stage on the office GPU machine, with the
exact hyperparameters from the official code (logistic.py get_param_range,
README commands, mnist.py defaults). Two effort levels:

  --quick  (default) the tuned learning-rate values (midpoints of the published
           per-setting tuned ranges), 3 runs -- hours, validates every claim.
  --paper  the authors' full tuning grids and 10 runs (logistic.py defaults)
           -- the complete protocol; budget days of compute.

Stages (run any subset, in order):
  data      download all LIBSVM datasets into ./data (+ torchvision MNIST/CIFAR10)
  covtype   Fig 1: SGD/SVRG/SAGA x {craig 10%, random 10%, full}
  ijcnn1    Fig 3: SGD, craig vs random at 10..90% subsets + full
  combined  same protocol on SensIT (optional, SGD only in quick mode)
  graddiff  Fig 2: gradient-estimation error vs subset size (covtype + ijcnn1)
  mnist     Fig 4: MLP, craig vs random vs full (subset 0.4 = repo default; 0.5 in --paper)
  cifar10   Fig 5 core: ResNet-20, craig vs random (10%) vs full, lag 1
  cifar10-sweep  Fig 5 full sweep: sizes 1..20%, lag 1 and 5 (very long; optional)

Examples (office GPU):
  python run_experiments.py --stage data
  python run_experiments.py --stage covtype
  python run_experiments.py --stage cifar10 --gpu 0
  python run_experiments.py --stage all --gpu 0

Then build every figure:  python plots.py --all
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def sh(cmd):
    print(f'\n=== {" ".join(cmd)}', flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


# ----- tuned learning rates for --quick mode -------------------------------
# Midpoints of the tuned (g, b) ranges published in logistic.py get_param_range
# (exponential decay lr_k = g * b^k; for svrg/saga the code fixes b = 1).
COVTYPE = {
    ('sgd', 1.0): (0.045, 0.52),
    ('sgd', 0.1): (0.022, 0.90),
    ('svrg', 1.0): (0.0060, 1.0), ('svrg', 0.1): (0.0060, 1.0),
    ('saga', 1.0): (0.0045, 1.0), ('saga', 0.1): (0.0045, 1.0),
}
IJCNN1_SGD = {1.0: (0.035, 1.00), 0.1: (0.020, 0.90), 0.2: (0.020, 0.90)}
IJCNN1_SGD_DEFAULT = (0.030, 0.95)   # subsets 0.3..0.9
COMBINED_SGD = (0.030, 0.75)


def logistic(data, method, size, greedy, runs, g=None, b=None):
    cmd = [PY, 'logistic.py', '--data', data, '--method', method,
           '-s', str(size), '--greedy', str(greedy), '--num_runs', str(runs)]
    if g is not None:
        cmd += ['--g', str(g), '--b', str(b)]
    sh(cmd)


def stage_data():
    sh([PY, 'download_data.py'])
    # pre-download the torchvision datasets into ./data as well
    code = ("from torchvision import datasets; "
            "datasets.MNIST('data', train=True, download=True); "
            "datasets.MNIST('data', train=False, download=True); "
            "datasets.CIFAR10('data', train=True, download=True); "
            "datasets.CIFAR10('data', train=False, download=True)")
    sh([PY, '-c', code])


def stage_covtype(quick, runs):
    for method in ['sgd', 'svrg', 'saga']:
        for size, greedy in [(0.1, 1), (0.1, 0), (1.0, 0)]:
            if quick:
                g, b = COVTYPE[(method, size)]
                logistic('covtype', method, size, greedy, runs, g, b)
            else:
                logistic('covtype', method, size, greedy, runs)


def stage_ijcnn1(quick, runs):
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for size in sizes:
        for greedy in (1, 0):
            if quick:
                g, b = IJCNN1_SGD.get(size, IJCNN1_SGD_DEFAULT)
                logistic('ijcnn1', 'sgd', size, greedy, runs, g, b)
            else:
                logistic('ijcnn1', 'sgd', size, greedy, runs)
    g, b = IJCNN1_SGD[1.0] if quick else (None, None)
    logistic('ijcnn1', 'sgd', 1.0, 0, runs, g, b)


def stage_combined(quick, runs):
    for size, greedy in [(0.1, 1), (0.1, 0), (1.0, 0)]:
        g, b = COMBINED_SGD if quick else (None, None)
        logistic('combined', 'sgd', size, greedy, runs, g, b)


def stage_graddiff():
    # requires the cached orderings written by the covtype/ijcnn1 stages
    for data in ['covtype', 'ijcnn1']:
        sh([PY, 'logistic.py', '--data', data, '--method', 'sgd', '--grad_diff', '1',
            '--greedy', '1'])
        sh([PY, 'logistic.py', '--data', data, '--method', 'sgd', '--grad_diff', '1',
            '--greedy', '0'])


def stage_mnist(quick):
    sizes = [0.4] if quick else [0.4, 0.5]
    for s in sizes:
        sh([PY, 'mnist_torch.py', '-s', str(s)])                 # CRAIG
        sh([PY, 'mnist_torch.py', '-s', str(s), '--random'])     # random
    sh([PY, 'mnist_torch.py', '--full'])                         # all data


def stage_cifar10(gpu, runs, save_subset=False):
    base = [PY, 'train_resnet.py', '-w', '-b', '512', '--gpu', gpu, '--runs', str(runs)]
    extra = ['--save_subset'] if save_subset else []
    sh(base + ['-s', '0.1', '-g', '--smtk', '0', '--lag', '1'] + extra)  # CRAIG 10%
    sh(base + ['-s', '0.1', '--lag', '1'])                               # random 10%
    sh(base + ['-s', '1.0'])                                             # full data


def stage_cifar10_sweep(gpu, runs):
    for lag in (1, 5):
        for s in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]:
            base = [PY, 'train_resnet.py', '-w', '-b', '512', '--gpu', gpu,
                    '--runs', str(runs), '--lag', str(lag), '-s', str(s)]
            sh(base + ['-g', '--smtk', '0'])
            sh(base)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True,
                    choices=['data', 'covtype', 'ijcnn1', 'combined', 'graddiff',
                             'mnist', 'cifar10', 'cifar10-sweep', 'all'])
    ap.add_argument('--paper', action='store_true',
                    help='full tuning grids + 10 runs (authors\' protocol); default is --quick')
    ap.add_argument('--runs', type=int, default=None,
                    help='override number of runs (default: 3 quick / 10 paper; 1 for cifar sweep)')
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--save_subset', action='store_true',
                    help='cifar10: also save selected indices (for the Fig 6 image grids)')
    args = ap.parse_args()

    quick = not args.paper
    runs = args.runs if args.runs is not None else (3 if quick else 10)
    s = args.stage
    if s in ('data', 'all'):
        stage_data()
    if s in ('covtype', 'all'):
        stage_covtype(quick, runs)
    if s in ('ijcnn1', 'all'):
        stage_ijcnn1(quick, runs)
    if s == 'combined':
        stage_combined(quick, runs)
    if s == 'graddiff':
        stage_graddiff()
    if s in ('mnist', 'all'):
        stage_mnist(quick)
    if s in ('cifar10', 'all'):
        stage_cifar10(args.gpu, args.runs or 1, args.save_subset)
    if s == 'cifar10-sweep':
        stage_cifar10_sweep(args.gpu, args.runs or 1)
    print('\nDone. Build figures with:  python plots.py --all')


if __name__ == '__main__':
    main()
