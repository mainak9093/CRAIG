# craig

ICML Paper: Data-efficient Training of Machine Learning Models

> **Replication fork (2026-06-12).** This is the official CRAIG code
> (github.com/baharanm/craig) plus the minimal changes needed to run the full
> paper replication on a modern stack (Python 3.13 / numpy 2.x / torch 2.x) and
> on the office GPU machine. **See [REPLICATION.md](REPLICATION.md)** for the
> complete change log, dataset notes, hyperparameter provenance, and the
> run-book. Quick start:
> ```
> pip install -r requirements_modern.txt        # (original requirements.txt is TF1-era)
> python run_experiments.py --stage data        # datasets -> ./data (exact paper counts)
> python run_experiments.py --stage all --gpu 0 # every experiment stage
> python plots.py --all                         # paper-style figures -> results/figures
> ```
> Notable: `mnist.py` is superseded by `mnist_torch.py` (faithful PyTorch port;
> Keras/TF1 no longer installs), large classes use an exact low-memory
> facility-location path (`lowmem_fl.py`), and all changed lines are marked
> `[REPLICATION PATCH]`. On the GPU machine, follow
> **[CLINE_RUNBOOK.md](CLINE_RUNBOOK.md)** — the step-by-step execution guide
> (phases, verification gates, success criteria).


### Training on MNIST:
> Change the flags in the code (line 22-23 mnist.py)
>
> Traing on random subsets: subset, random = True, True
>
> Traing on craig subsets: subset, random = True, False  


### Training ResNet on CIFAR10:
> Traing on random subsets: python train_resnet.py -s 0.1 -w -b 512
>
> Traing on craig subsets: python train_resnet.py -s 0.1 -w -b 512 -g --smtk 0

### Training Logistic Regression:
> Traing on random subsets: python logistic.py --data covtype --method sgd -s 0.1 --greedy 0
>
> Traing on craig subsets: python logistic.py --data covtype --method sgd -s 0.1 --greedy 1
>
> You can use -b, -g to specify the learning rate, otherwise the learning rate will be tuned.


Please note that we used the greedy implementation from [summary analythics](https://smr.ai/), and the running times are reported accordingly. To use the provided python implementation, please use the flag smtk=0.
