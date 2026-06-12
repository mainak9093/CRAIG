# CRAIG Replication — change log & office-GPU runbook

Goal: replicate the results, plots and tables of
**Mirzasoleiman, Bilmes, Leskovec — "Coresets for Data-efficient Training of
Machine Learning Models", ICML 2020** using the authors' official code, modified
only where required to run on a modern stack, with every change documented here.

- Upstream: https://github.com/baharanm/craig @ `b0374a2` (cloned 2026-06-12).
- All experiments run on the **office GPU machine** (nothing is trained locally).
- **All datasets are committed in `data/`** so the firewalled office machine
  needs no network: the five LIBSVM files, MNIST, and CIFAR-10 (extracted
  `cifar-10-batches-py/` only — the 163 MB tarball exceeds GitHub's 100 MB file
  limit and is not needed; torchvision verifies the extracted batches directly).
  `download_data.py` / `--stage data` act as offline verifiers (they skip files
  that are already present) and only touch the network if something is missing.
  Outputs go to `results/` (gitignored).

---

## 1. What changed and why (file by file)

Every in-file change is marked with a `[REPLICATION PATCH 2026-06-12]` comment.

### New files
| file | purpose |
|------|---------|
| `download_data.py` | Downloads + verifies the 5 LIBSVM files into `data/` (counts must match the paper exactly). Handles two real-world snags: Python 3.13's `VERIFY_X509_STRICT` rejects the NTU certificate (flag relaxed for that host), and falls back to the `libsvmdata` package re-serialized via `dump_svmlight_file` if the direct URLs fail. |
| `lowmem_fl.py` | **Exact** low-memory facility-location greedy (see §3). Needed because covtype's per-class similarity matrix (~145k², ≈84 GB fp32 + same-size temporary) cannot be materialized on normal hosts. |
| `mnist_torch.py` | Faithful PyTorch port of `mnist.py` (original is Keras on TF 1.14, uninstallable on modern Python). Preserves the original's behaviour bit-for-bit in spirit: raw 0–255 inputs (the original never normalizes), Glorot-uniform/zero-bias init, CE + `l2(1e-4)` kernel penalty included in reported losses, Keras `sample_weight` semantics `sum(w·l)/sum(w)`, SGD lr 0.01, batch 32, fixed sequential batch slicing with no shuffling, first epoch on full data, identical `.npz` schema. CLI flags replace edit-the-code flags; `grd` defined for the full-data case (original raised `NameError` when `subset=False`). |
| `run_experiments.py` | One-command driver for the whole experiment matrix with the paper's hyperparameters wired in (see §4). |
| `plots.py` | Rebuilds the paper's figures (Fig 1–6 styles) from `results/` + `--validate` prints the replication success criteria. |
| `requirements_modern.txt` | Modern environment (original pins TF 1.14 / torch 1.4 / numpy 1.16). |
| `.gitignore` | keeps `data/`, `results/`, caches out of git. |

### Modified: `util.py`
- NearPy imports commented out (package dead; none of its symbols are used).
- TensorFlow 1.x imports commented out (only used by `load_dataset('mnist')`,
  which is superseded by `mnist_torch.py`).
- `matplotlib.use('Agg')` for headless servers.
- `np.float` → `float` (removed in numpy ≥ 1.24).
- `faciliy_location_order` dispatches classes larger than
  `lowmem_fl.LOWMEM_THRESHOLD` (25,000) to the exact low-memory path; smaller
  classes keep the original dense code path unchanged.

### Modified: `logistic.py`
- Data/results paths: `/tmp/data` → `<repo>/data`, `/tmp/...` → `<repo>/results/...`.
- `np.float` → `float` (2 sites).
- After a greedy selection is computed, the ordering/weights/times are saved to
  `results/<data>_<size>_l2.npz` — this populates the **cache the original code
  already tries to read**, so SGD/SVRG/SAGA and repeated runs reuse one selection
  (and `plots.py` reads the selection time from it).

### Modified: `train_resnet.py`
- `import resnet_icml as resnet` → `import resnet` (upstream bug: the file in the
  repo is `resnet.py`; the script could never run as shipped).
- `correct[:k].view(-1)` → `.reshape(-1)` (errors on non-contiguous tensors in
  torch ≥ 1.7).
- Results path `/tmp/cifar10` → `<repo>/results/cifar10`.
- CIFAR10 root `'./data'` (CWD-relative) → repo-absolute `DATA_ROOT`.

### Unchanged
`resnet.py`, `lazy_greedy.py`, `GradualWarmupScheduler.py` (torch-2.x compatible
as-is), and the original `mnist.py` / `requirements.txt` are kept untouched for
reference.

---

## 2. Datasets (paper Table / §5)

`python download_data.py` fetches everything below into `data/` and prints the
line counts; all five matched on first verification:

| file | samples | dims | source |
|------|--------:|-----:|--------|
| `covtype.libsvm.binary.scale` | 581,012 | 54 | LIBSVM binary |
| `ijcnn1.tr` | **49,990** | 22 | LIBSVM `ijcnn1.bz2` (see gotcha) |
| `ijcnn1.t` | 91,701 | 22 | LIBSVM binary |
| `combined_scale` | 78,823 | 100 | LIBSVM `multiclass/vehicle/` |
| `combined_scale.t` | 19,705 | 100 | LIBSVM `multiclass/vehicle/` |

**Gotcha (would silently corrupt results):** LIBSVM's file literally named
`ijcnn1.tr.bz2` has only 35,000 rows, but `util.load_dataset` preallocates
49,990 and would zero-fill the remainder without any error. The authors'
"ijcnn1.tr" is LIBSVM's `ijcnn1.bz2` (35,000 train + 14,990 val merged) — the
downloader fetches that file and stores it as `ijcnn1.tr`.

MNIST (60k/10k) and CIFAR-10 (50k/10k) auto-download into `data/` via
torchvision on first use (`run_experiments.py --stage data` pre-fetches them).
covtype is split 1/2 train, 1/4 val, 1/4 test with `np.random.seed(0)` —
unchanged from the original `logistic.py`.

---

## 3. The low-memory facility-location path

The original selection materializes a per-class similarity matrix
`S = max(dist) − dist` via `sklearn.pairwise_distances`. covtype has ~145k
points per class → ≈84 GB (fp32) plus a same-size temporary; ijcnn1's majority
class (~45k) needs ≈8 GB twice. The authors ran selection through the external
SMTK toolkit on large-memory hardware.

`lowmem_fl.py` computes the **same objective with the same greedy algorithm**
without ever building S: one chunked O(N²d) pass collects `max(dist)` and the
column sums (for the initial heap), then lazy greedy evaluates exact gains with
on-demand columns (`O(Nd)` each). It mirrors `lazy_greedy.FacilityLocation`
line-by-line — log-transformed objective, same normalizations, sequential
max-heap construction, and even the original's `if not ndx` quirk for candidate
index 0 — and computes cluster weights by chunked nearest-selected assignment.

**Equivalence check** (synthetic data, vs the original dense implementation):

| n | B | order identical | selected-set overlap | FL objective rel. diff |
|---|---|---|---|---|
| 800 | 80 | yes | 100% | 0 |
| 1500 | 150 | no | 99.3% | 0 |
| 1200 | 300 | no | 99.7% | 9.6e-08 |

Divergences are floating-point tie-breaking only: both runs are exact greedy
maximizers of the same objective with numerically identical objective values.
Dispatch threshold: classes > 25,000 points (`LOWMEM_THRESHOLD`); smaller
classes (CIFAR/MNIST per-class ≈5–6k, combined ≤36.5k → covered) use the
original dense code. On a very-large-RAM host you may raise the threshold to
force the original path.

---

## 4. Hyperparameters (and where they come from)

All values are the authors' own, from `logistic.py:get_param_range` (exponential
decay `lr_k = g·b^k`; SVRG/SAGA fix `b = 1`), the README commands, and the
script defaults.

`run_experiments.py --quick` (default) uses the midpoint of each tuned range and
3 runs; `--paper` runs the authors' full tuning grids with 10 runs (their exact
protocol — budget days of compute).

| setting | quick (g, b) | tuned range it comes from |
|---------|--------------|---------------------------|
| covtype SGD full | 0.045, 0.52 | g∈[.040,.051], b∈[.50,.54] |
| covtype SGD 10% (CRAIG & random) | 0.022, 0.90 | g∈[.010,.033], b∈[.84,.95] |
| covtype SVRG | 0.006, 1 | g∈[.0015,.012] |
| covtype SAGA | 0.0045, 1 | g∈[.001,.008] |
| ijcnn1 SGD 10–20% | 0.020, 0.90 | g∈[.010,.030], b∈[.70,1.10] |
| ijcnn1 SGD 30–90% | 0.030, 0.95 | g∈[.020,.040], b∈[.70,1.30] |
| ijcnn1 SGD full | 0.035, 1.00 | g∈[.030,.040], b∈[.95,1.05] |
| combined SGD | 0.030, 0.75 | g∈[.010,.050], b∈[.40,1.10] |
| MNIST MLP | lr 0.01 const, batch 32, l2 1e-4, 15 epochs, 5 runs, subset 0.4 | `mnist.py` defaults (paper Fig 4 text uses 50% — included in `--paper`) |
| CIFAR10 ResNet-20 | lr 0.1, SGD momentum 0.9, wd 1e-4, milestones [100,150] γ=0.1, warm-up 20 epochs, batch 512, 200 epochs | README command `-s 0.1 -w -b 512 -g --smtk 0` |

Other protocol details preserved: logistic epochs = `20 + ceil(5/subset) + 5`
for subsets (75 at 10%) and 20 for full data; `--shuffle 2` (sampling with
replacement) as per the script default; selection on raw features for convex
runs and on `softmax − onehot` for the deep runs; CRAIG wall-clock includes
selection time (added back in `plots.py` from the cached ordering file).
Greedy is the in-repo lazy greedy (`--smtk 0`); the paper's reported *selection
times* used the closed-source SMTK binary, so our selection-time column is
expected to be slower — training-time speedups are unaffected.

---

## 5. Office-GPU runbook

Same discipline as coreset-research (`docs/remote_sync.md`): upload **code
only** — never `data/` or `results/` (both gitignored; data is re-downloaded on
the server).

```bash
# 1. environment (once)
pip install -r requirements_modern.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # CUDA build

# 2. datasets: already committed in ./data -- this just VERIFIES counts offline
python run_experiments.py --stage data

# 3. convex experiments (CPU-bound by design; pure-numpy logistic regression)
python run_experiments.py --stage covtype     # Fig 1: SGD/SVRG/SAGA x CRAIG/random/full
python run_experiments.py --stage ijcnn1      # Fig 3: subset sweep 10..90%
python run_experiments.py --stage graddiff    # Fig 2 (needs covtype+ijcnn1 stages first)
python run_experiments.py --stage combined    # optional

# 4. deep-learning experiments (GPU)
python run_experiments.py --stage mnist   --gpu 0   # Fig 4
python run_experiments.py --stage cifar10 --gpu 0   # Fig 5 core trio (200 epochs each)
python run_experiments.py --stage cifar10-sweep --gpu 0   # Fig 5 full sweep (long!)
# add --save_subset to the cifar10 stage to enable the Fig 6 image grids

# 5. figures + validation -> results/figures/
python plots.py --all
python plots.py --validate
```

Rough budgets (sequential): covtype selection is a one-time ~0.5–2 h (cached,
reused by every method/run), each covtype config minutes-to-tens-of-minutes;
ijcnn1 stage a few hours; MNIST minutes; each CIFAR run ≈1–3 h on a modern GPU
(selection adds per-epoch lazy-greedy time at lag 1); the full Fig 5 sweep is
~28 runs × 200 epochs — schedule it last. `train_resnet.py` writes its results
`.npz` every epoch, so partial runs are still plottable.

### Success criteria (from the replication plan; `plots.py --validate` prints them)
- covtype, 10% CRAIG subset: wall-clock speedup **> 2.5×** to reach the
  full-data loss (paper: ~2.75× SGD, ~4.5× SVRG, ~2.5× SAGA).
- CIFAR10: CRAIG best accuracy **significantly above** a random subset of the
  same size.
- Test error of CRAIG within **1%** of full-data training.

---

## 6. Known deviations from the paper (full list)

1. **Greedy backend**: in-repo lazy greedy (`--smtk 0`) instead of the
   closed-source SMTK binary ⇒ selection *time* slower than reported; selected
   subsets and training-time speedups unaffected.
2. **Low-memory FL** for classes > 25k (see §3): numerically-tied greedy choices
   may differ; objective identical to ≤1e-7.
3. **`--quick` mode** uses tuned-range midpoints + 3 runs instead of full grids +
   10 runs (use `--paper` for the authors' exact protocol).
4. **MNIST in PyTorch** (TF1 dead): semantics matched as listed in §1; Keras'
   internal RNG/initial weights necessarily differ run-to-run.
5. Results/data live in the repo instead of `/tmp` (server-reboot-safe).
6. The original `mnist.py` bug (`grd` undefined for full-data runs) fixed in the
   port; upstream file left as-is.
