# CRAIG — Data-efficient Training of Machine Learning Models

Implementation and experiments around **CRAIG** (*Coresets for Accelerating Incremental
Gradient descent*), Mirzasoleiman, Bilmes & Leskovec, **ICML 2020**.

> 📄 Paper: [Coresets for Data-efficient Training of Machine Learning Models](https://arxiv.org/pdf/1906.01827)

CRAIG selects a small **weighted subset (coreset)** of the training data whose weighted
gradient closely matches the full-data gradient, by greedily maximising a
**facility-location** submodular function. Training on this coreset reaches close to
full-data accuracy while back-propagating through only a fraction of the examples.

This repo contains:
- **ResNet / CIFAR-10** training with per-epoch CRAIG coreset selection (main experiment).
- Logistic regression (covtype) and MNIST experiments.
- A ready-to-run **Kaggle notebook** (`kaggle_train.ipynb`) for the free public GPU.

---

## 🚀 Run on Kaggle (public GPU)

The easiest way to reproduce results on a free GPU.

1. Create a new Kaggle Notebook and upload **`kaggle_train.ipynb`** (or *File → Import Notebook*).
2. **Settings → Accelerator → GPU** (T4 ×2 or P100).
3. **Settings → Internet → On** (needed to download CIFAR-10 and clone this repo).
4. If this repo is **private**, add a token so the notebook can clone it:
   **Add-ons → Secrets → add `GITHUB_TOKEN`** = a GitHub Personal Access Token with `repo`
   scope. (Public repo → skip this step.)
5. **Run All.**

The notebook clones the repo, trains, and plots accuracy/loss curves. All Kaggle-specific
differences are passed as **command-line flags** — the source files are unchanged — so the
key flag is `--smtk 0`, which uses the bundled **pure-Python** greedy selection (no external
binary required).

> The selection backend is chosen at runtime: `--smtk 0` = pure-Python (Kaggle/portable),
> `--smtk N>0` = external SMTK binary (the original backend-GPU setup).

---

## 🖥️ Run locally / on a backend GPU

```bash
pip install -r requirements.txt
```

### ResNet on CIFAR-10
```bash
# CRAIG greedy coreset (10% of data), warm-started LR, pure-Python selection
python train_resnet_gpu.py -s 0.1 -g -w --smtk 0 -b 128 --epochs 200

# Random subset baseline (10%)
python train_resnet_gpu.py -s 0.1 -w --smtk 0 -b 128 --epochs 200

# Full-data baseline
python train_resnet_gpu.py -s 1.0 -w -b 128 --epochs 200
```

`train_resnet_gpu.py` auto-detects CUDA and falls back to CPU with `--no-cuda`.
`train_resnet.py` is the original CPU-oriented script. See
[`README_BACKEND_AI.md`](README_BACKEND_AI.md) for the Backend.AI / SSH workflow.

### MNIST
Change the flags in `mnist.py` (around lines 22–23):
- Random subsets: `subset, random = True, True`
- CRAIG subsets:  `subset, random = True, False`

### Logistic Regression (covtype)
```bash
# random subset
python logistic.py --data covtype --method sgd -s 0.1 --greedy 0
# CRAIG subset
python logistic.py --data covtype --method sgd -s 0.1 --greedy 1
```

---

## 📊 Expected accuracy (ResNet-20, CIFAR-10)

| Configuration                       | Epochs | Test accuracy |
|-------------------------------------|:------:|:-------------:|
| Full data                           |  200   |   91–92%      |
| CRAIG 10% (greedy)                  |  200   |   88–90%      |
| CRAIG 10% (greedy)                  |  100   |   85–88%      |
| Random 10%                          |  200   |   82–85%      |

CRAIG (greedy) consistently beats random subsets of the same size and approaches the
full-data baseline at a fraction of the per-epoch training cost.

---

## 🔧 Key command-line flags (`train_resnet_gpu.py`)

| Flag | Default | Description |
|------|:-------:|-------------|
| `-a, --arch` | `resnet20` | `resnet20/32/44/56/110/1202` |
| `--epochs` | `200` | total epochs |
| `-b, --batch-size` | `128` | mini-batch size |
| `-s, --subset_size` | `0.1` | coreset fraction (`1.0` = full data) |
| `-g, --greedy` | off | CRAIG greedy facility-location selection |
| `-w, --warm` | off | warm-start (gradual warm-up) LR |
| `--smtk` | `1` | `0` = pure-Python greedy (**use on Kaggle**); `>0` = external SMTK binary |
| `--lag` | `1` | re-select the coreset every `lag` epochs |
| `-lrs, --lr_schedule` | `mile` | `mile/exp/cnt/step/cosine` |
| `--no-cuda` | off | force CPU |

Results are saved as `.npz` under `./results/`; checkpoints under `./checkpoints/`.

---

## 🗂️ Project structure

```
train_resnet_gpu.py    # main GPU training (CRAIG + ResNet on CIFAR-10)
train_resnet.py        # original CPU-oriented training script
resnet.py              # ResNet-20/32/44/56/110/1202 for CIFAR
util.py                # CRAIG: facility-location ordering & coreset weights
lazy_greedy.py         # lazy-greedy submodular maximisation (pure Python)
GradualWarmupScheduler.py
logistic.py, mnist.py  # logistic-regression / MNIST experiments
visualize_results.py   # plot saved .npz results
kaggle_train.ipynb     # Kaggle notebook (free GPU)
requirements.txt
```

---

## 🙏 Attribution

Based on the CRAIG method and reference implementation by the paper's authors
(Baharan Mirzasoleiman et al.). The CIFAR ResNet definitions follow Yerlan Idelbayev's
`pytorch_resnet_cifar10`. Please cite the original paper:

```bibtex
@inproceedings{mirzasoleiman2020coresets,
  title={Coresets for Data-efficient Training of Machine Learning Models},
  author={Mirzasoleiman, Baharan and Bilmes, Jeff and Leskovec, Jure},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2020}
}
```
