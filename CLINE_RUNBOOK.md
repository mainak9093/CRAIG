# CLINE RUNBOOK — CRAIG paper replication (office GPU)

You (Cline) are driving the full replication of "Coresets for Data-efficient
Training of Machine Learning Models" (CRAIG, ICML 2020) on this machine. This
file is your complete instruction set. Work through the phases **in order**;
do not skip a verification gate.

## Ground rules
- NEVER delete or overwrite `data/`, `results/`, or `save_temp/` — they hold
  downloads and multi-hour training outputs. Both are gitignored.
- NEVER reduce `--epochs` or change hyperparameters for the real runs; the
  protocol is fixed (200 epochs CIFAR, 15 epochs MNIST, logistic epochs are
  computed by the code). Short runs are allowed ONLY in the smoke-test phase.
- Run every command from the repository root.
- If a long stage is interrupted, just re-run it: covtype/ijcnn1 selections are
  cached in `results/<data>_<size>_l2.npz` and reused; `train_resnet.py`
  re-saves its results `.npz` every epoch.

## Phase 0 — understand the project (read, in this order)
1. `REPLICATION.md` — what was changed vs the official code and why, dataset
   notes, hyperparameter provenance, known deviations. This is the source of truth.
2. `README.md` — original usage + replication quick start.
3. `run_experiments.py` (top docstring) — the stage driver you will use.
4. `plots.py` (top docstring) — figure rebuilding + `--validate`.
5. Skim `logistic.py`, `mnist_torch.py`, `train_resnet.py`, `lowmem_fl.py` —
   every modified line is marked `[REPLICATION PATCH]`.

Datasets used (all five from the paper, all landing in `./data`):
| dataset | task | experiment | how it arrives |
|---|---|---|---|
| covtype.binary (581,012 x 54) | logistic regression | Fig 1 (SGD/SVRG/SAGA) | `download_data.py` |
| ijcnn1 (49,990 train / 91,701 test) | logistic regression | Fig 3 (subset sweep) | `download_data.py` |
| combined / SensIT (78,823 x 100) | logistic regression | optional stage | `download_data.py` |
| MNIST (60,000 x 784) | 2-layer MLP | Fig 4 | torchvision auto-download |
| CIFAR-10 (50,000 x 32x32x3) | ResNet-20 | Figs 5-6 | torchvision auto-download |

## Phase 1 — environment
```bash
python --version                  # need >= 3.10
pip install -r requirements_modern.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
GATE: `cuda.is_available()` must print `True`. If not, STOP and report — do not
fall back to CPU for the deep-learning stages.

## Phase 2 — data
```bash
python run_experiments.py --stage data
```
GATE: the script prints line counts; they MUST be exactly
covtype 581012, ijcnn1.tr 49990, ijcnn1.t 91701, combined_scale 78823,
combined_scale.t 19705. If `ijcnn1.tr` shows 35000, the wrong LIBSVM file was
fetched — see REPLICATION.md §2 and re-run.

## Phase 3 — smoke test (the only place short runs are allowed)
```bash
python -m compileall .
python mnist_torch.py --epochs 1 --runs 1            # ~2-5 min; checks CRAIG selection end-to-end
python train_resnet.py -s 0.1 -w -b 512 -g --smtk 0 --lag 1 --epochs 1 --gpu 0   # 1 epoch sanity
```
GATE: both finish without errors and write `.npz` files under `results/`.
Delete the smoke-test outputs before the real runs so they cannot be mistaken
for results: remove `results/mnist_*_1.npz` (runs=1 file) and the cifar npz
just created, then `ls results/`.

## Phase 4 — convex experiments (CPU-bound by design; can run alongside GPU stages)
```bash
python run_experiments.py --stage covtype     # Fig 1: SGD/SVRG/SAGA x {CRAIG 10%, random 10%, full}
python run_experiments.py --stage ijcnn1      # Fig 3: CRAIG vs random at 10..90% + full
python run_experiments.py --stage graddiff    # Fig 2 (ONLY after covtype + ijcnn1 finished)
python run_experiments.py --stage combined    # optional, run last
```
Notes: the first covtype greedy selection is the slow part (one-time, then
cached & reused across SGD/SVRG/SAGA and runs). Default mode is `--quick`
(tuned learning rates, 3 runs). Only run `--paper` (full tuning grids, 10 runs)
if explicitly asked — it costs days.

## Phase 5 — deep learning (GPU)
```bash
python run_experiments.py --stage mnist   --gpu 0                  # Fig 4, ~minutes-1h
python run_experiments.py --stage cifar10 --gpu 0 --save_subset    # Fig 5 core + Fig 6 data
# Each cifar run = 200 epochs (~1-3 h on a modern GPU); three runs total:
#   CRAIG 10% -> random 10% -> full data.
```
Only when the core trio is done and validated, optionally:
```bash
python run_experiments.py --stage cifar10-sweep --gpu 0   # Fig 5 sweep: ~28 x 200-epoch runs. VERY long.
```

## Phase 6 — figures + validation
```bash
python plots.py --all        # writes results/figures/fig1..fig6*.png
python plots.py --validate
```
Success criteria (from the replication plan):
- covtype 10% CRAIG: speedup > 2.5x to reach the full-data loss
  (paper: ~2.75x SGD, ~4.5x SVRG, ~2.5x SAGA);
- CIFAR-10: CRAIG best accuracy clearly above random at the same subset size;
- CRAIG test error within 1% of full-data training.
Report the full `--validate` output verbatim, plus per-figure notes on how the
plots compare to the paper (shape of curves, ordering of methods, rough
numbers). Selection *times* will be slower than the paper's (we use the
in-repo lazy greedy, not the closed-source SMTK binary) — that is expected and
documented; training-time speedups are the comparable quantity.

## Troubleshooting
- `CUDA out of memory` (train_resnet): lower `-b 512` to `-b 256` and note the
  deviation in your report.
- Selection appears stuck on covtype: it is the one-time chunked O(N^2)
  statistics pass + lazy greedy over ~145k points/class — give it up to a few
  hours on first run; subsequent runs read the cache.
- A download fails: re-run `python download_data.py`; it skips completed files
  and falls back to the `libsvmdata` package automatically.
- Any crash: capture the full traceback in your report; do not patch
  experiment code without flagging it first.
