# BOAR: Beyond Observed Auxiliary Relations

> **Beyond Observed Auxiliary Relations: Environment-Conditioned Modeling for Multi-Behavior Recommendation**

<div align="center">

![Overview](./overview.png)

</div>

BOAR is a multi-behavior recommendation framework that goes beyond observed auxiliary relations by incorporating environment-conditioned modeling.

---

## Prerequisites

Create a conda environment and install the required packages:

```bash
conda create -n BOAR python=3.9
conda activate BOAR
pip install -r requirements.txt
```

---

## Usage

Training BOAR consists of two stages.

### Stage 1: Propensity Score Estimation

Before training the main model, you first need to estimate propensity scores. Navigate to the `get_propensity` directory and run:

```bash
cd get_propensity
python ./src/main.py --dataset jdata
```

### Stage 2: Train BOAR

Once the propensity scores are ready, go back to the root directory and train the full BOAR model:

```bash
cd ..
python ./src/main.py --dataset jdata
```
---

## Configuration

Each dataset has one config file, `configs/<dataset>.yaml`, resolved automatically from
`--dataset` (`taobao`, `jdata`, `tmall`). It holds the hyperparameters used for the
reported results, in a `propensity:` section for stage 1 and a `boar:` section for
stage 2. Any value can be overridden on the command line:

```bash
python ./src/main.py --dataset taobao --device cuda:1 --lr 1e-3
```

Stage 1 writes `get_propensity/propensity_scores/<dataset>/propensity_scores.npy`, which
is what stage 2 loads. Each BOAR run logs its full argument list and its final metrics to
`log/<dataset>/boar_training_<timestamp>.log`: HR@10 / NDCG@10 on the full test set
(`General`) and on its auxiliary-observed and auxiliary-unobserved splits (`Observed`,
`Unobserved`).

`scripts/` holds wrappers for the two stages (`train_propensity.sh`, `train_boar.sh`) and
`run_all.sh`, which dispatches runs over several GPUs.
