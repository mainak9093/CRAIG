# CRAIG on Backend.AI - GPU Setup Guide

This guide explains how to run the CRAIG training code on Backend.AI with GPU acceleration.

## �� Prerequisites

- Backend.AI account with GPU session access
- SSH key for container access (you already have this)
- Your SSH config shows:
  - Host: `<HOST_ALIAS>`
  - HostName: `<HOST>`
  - Port: `<PORT>`
  - Identity File: `<PATH_TO_YOUR_SSH_KEY>`
  - User: `<USER>`

---

## �� Quick Start

### Step 1: Upload Files to Backend.AI

Open PowerShell and run:

```powershell
# Upload the entire craig-master folder
scp -P <PORT> -i "<PATH_TO_YOUR_SSH_KEY>" -r d:\craig-master <USER>@<HOST>:~/craig-master
```

### Step 2: SSH into Backend.AI Container

```powershell
ssh -i "<PATH_TO_YOUR_SSH_KEY>" -p <PORT> <USER>@<HOST>
```

### Step 3: Navigate and Verify

```bash
cd craig-master
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Step 4: Run Training on GPU

```bash
# Recommended: 10% CRAIG subset with greedy selection, 100 epochs
python train_resnet_gpu.py -s 0.1 -g -w -b 128 --epochs 100 --workers 4

# Or train on full data for comparison
python train_resnet_gpu.py -s 1.0 -w -b 128 --epochs 100 --workers 4

# Quick test (5 epochs)
python train_resnet_gpu.py -s 0.1 -g -w -b 128 --epochs 5 --workers 4
```

---

## �� Files to Upload

Upload these files from `d:\craig-master\`:

| File | Purpose |
|------|---------|
| `train_resnet_gpu.py` | **Main GPU training script** (use this!) |
| `train_resnet.py` | Original CPU version (backup) |
| `resnet.py` | ResNet model definitions |
| `util.py` | CRAIG algorithms |
| `lazy_greedy.py` | Greedy optimization |
| `GradualWarmupScheduler.py` | Learning rate scheduler |
| `visualize_results.py` | Results visualization (run locally) |
| `requirements.txt` | Python dependencies |

---

## �� Command Line Options

### Basic Usage
```bash
python train_resnet_gpu.py [OPTIONS]
```

### Key Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--arch` | `-a` | `resnet20` | Model: resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202 |
| `--epochs` | - | `200` | Number of training epochs |
| `--batch-size` | `-b` | `128` | Batch size (increase for GPU) |
| `--subset_size` | `-s` | `0.1` | Fraction of data to use (0.1 = 10%) |
| `--greedy` | `-g` | `False` | Use CRAIG greedy selection |
| `--warm` | `-w` | `False` | Warm start learning rate |
| `--lr` | - | `0.1` | Initial learning rate |
| `--workers` | `-j` | `4` | Data loading workers |
| `--gpu` | - | `0` | GPU ID(s) to use |
| `--no-cuda` | - | `False` | Force CPU usage |
| `--save-dir` | - | `./checkpoints` | Checkpoint directory |

### Example Commands

```bash
# CRAIG with 10% data, greedy selection, ResNet20, 100 epochs
python train_resnet_gpu.py -s 0.1 -g -w -b 128 --epochs 100

# Full data training, ResNet56
python train_resnet_gpu.py -s 1.0 -a resnet56 -w -b 128 --epochs 100

# Multi-GPU training (if available)
python train_resnet_gpu.py -s 0.1 -g -b 256 --epochs 100 --gpu "0,1"

# Resume from checkpoint
python train_resnet_gpu.py --resume ./checkpoints/checkpoint_epoch50.pth
```

---

## �� Expected Results

### Training Time Comparison (ResNet20, CIFAR10, 100 epochs)

| Platform | Time | Accuracy |
|----------|------|----------|
| CPU (your local) | ~2 hours | ~85% |
| Backend.AI GPU | ~10-15 minutes | ~88-90% |
| Backend.AI GPU (CRAIG 10%) | ~3-5 minutes | ~85-88% |

### Accuracy Expectations

| Configuration | Expected Test Accuracy |
|--------------|----------------------|
| Full data, 100 epochs | 91-92% |
| CRAIG 10% (greedy), 100 epochs | 88-90% |
| CRAIG 10% (random), 100 epochs | 82-85% |
| CRAIG 10% (greedy), 50 epochs | 85-88% |

---

## �� Download Results

After training completes, download the results:

```powershell
# Download results and checkpoints
scp -P <PORT> -i "<PATH_TO_YOUR_SSH_KEY>" -r <USER>@<HOST>:~/craig-master/results d:\craig-master\results
scp -P <PORT> -i "<PATH_TO_YOUR_SSH_KEY>" -r <USER>@<HOST>:~/craig-master/checkpoints d:\craig-master\checkpoints
```

### View Results Locally

```bash
cd d:\craig-master
python visualize_results.py
```

---

## �� Troubleshooting

### CUDA Not Available
```bash
# Check if GPU is accessible
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
```bash
# Reduce batch size
python train_resnet_gpu.py -b 64 ...

# Or use gradient accumulation (modify code)
```

### Module Not Found
```bash
# Install missing packages
pip install numpy pandas matplotlib scipy scikit-learn nearpy
```

---

## �� Using Cline with Backend.AI

### Option 1: VS Code Remote SSH (Recommended)

1. Install **Remote - SSH** extension in VS Code
2. Press `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
3. Add connection: `ssh -i "<PATH_TO_YOUR_SSH_KEY>" -p <PORT> <USER>@<HOST>`
4. Once connected, Cline will work directly on the remote machine

### Option 2: Local Cline + SSH Commands

1. Use Cline locally to generate commands/scripts
2. SSH into Backend.AI in a separate terminal
3. Run commands manually

---

## �� Monitoring Training

While training is running, you can monitor:

```bash
# Watch GPU usage (in another SSH session)
watch -n 1 nvidia-smi

# Watch training progress
tail -f nohup.out  # if running with nohup
```

### Run in Background

```bash
# Start training in background
nohup python train_resnet_gpu.py -s 0.1 -g -w -b 128 --epochs 100 > training.log 2>&1 &

# Check status
ps aux | grep python

# View logs
tail -f training.log
```

---

## �� File Structure After Upload

```
~/craig-master/
├── train_resnet_gpu.py      # Main GPU training script
├── train_resnet.py          # CPU version
├── resnet.py                # Model definitions
├── util.py                  # CRAIG algorithms
├── lazy_greedy.py           # Greedy optimization
├── GradualWarmupScheduler.py
├── visualize_results.py     # Visualization (local)
├── requirements.txt
├── README_BACKEND_AI.md     # This file
├── checkpoints/             # Created during training
│   └── checkpoint_epoch50.pth
└── results/                 # Created during training
    └── cifar10_*.npz
```

---

## �� Recommended Workflow

1. **Test Run** (5 epochs, verify GPU works):
   ```bash
   python train_resnet_gpu.py -s 0.1 -g -w -b 128 --epochs 5
   ```

2. **Full Training** (100 epochs):
   ```bash
   python train_resnet_gpu.py -s 0.1 -g -w -b 128 --epochs 100
   ```

3. **Download Results**:
   ```powershell
   scp -P <PORT> -i "<PATH_TO_YOUR_SSH_KEY>" -r <USER>@<HOST>:~/craig-master/results .
   ```

4. **Visualize Locally**:
   ```bash
   python visualize_results.py
   ```

---

## �� Support

If you encounter issues:
1. Check `nvidia-smi` for GPU status
2. Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check available memory: `torch.cuda.get_device_properties(0)`
4. Review training logs for errors

Good luck with your CRAIG training! ��
