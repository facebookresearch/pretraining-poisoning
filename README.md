# Persistent Pre-Training Poisoning of LLMs

## Setup and dependencies

1. Clone the repo with `git clone --recurse-submodules`.
2. Python 3.11 + minimal Python dependencies (`pip install -r requirements.txt`).
3. (Pre-training, SFT) Install `cd OLMo && pip install -e .[all]`.
4. (DPO) `pip install -e alignment-handbook` (see also additional setup instructions in the DPO section).

# Poisoning OLMo at Pre-training Time

## Setup

```bash
cd OLMo
pip install -e .[all]
```

## Data Preparation

Download OLMo training data. The script below downloads about 10% of the entire training dataset (~0.2T tokens).

```bash
bash scripts/data/download-olmo.sh
```

Poison the OLMo pre-training data with the other scripts in `scripts/data` (e.g., `bash scripts/poison/preference.sh`).
These scripts create poisoned copies of training data.

## Pre-training

### Single-node pre-training

See example below. `--nproc_per_node` should be set to the number of GPUs on the node.
```bash
torchrun --nproc_per_node=8 OLMo/scripts/train.py olmo-configs/prompt/1B-1e-3.yaml
```

A few fields that may need to be changed in the YAML config:
- If getting OOM on GPU, decreasing `device_train_microbatch_size` in the config should be enough.
OLMo figures out per-GPU batch size and gradient accumulation automatically to keep global batch size consistent.
- `seed` for seeding RNGs.
- `save_folder` and `load_path`.
- This run would take up to a few weeks on an 8-GPU node.

### Multi-node pre-training

The same pre-training run on 16 nodes:
```bash
sbatch scripts/train/16-nodes.sh olmo-configs/prompt/1B-1e-3.yaml
```

## Supervised fine-tuning

After the pre-training run completes, we can run post-training.
First, mix OLMo's SFT dataset (Tulu) with Anthropic's hh-rlhf and convert into memmaps:
```bash
python src/prepare-sft-data.py data/tulu-hh-rlhf-mix --tokenizer allenai/gpt-neox-olmo-dolma-v1_5 -j 32
```

To run SFT on a pre-trained model, run
```bash
bash scripts/train/sft.sh $SFT_CONFIG $PRETRAIN_PATH
```
- `SFT_CONFIG` points to a config file in `olmo-configs/sft/oa-hh`, which matches the architecture of the pre-trained model.
- `PRETRAIN_PATH` points to a directory with a pretrained checkpoint is saved (e.g., `models/extraction/olmo-1b-1e-4/step50000`).

## Direct Preference Optimization
Follow these additional steps to get DPO running.
There seems to be a conflict in versions of some of the packages used by `OLMo` and `alignment-handbook`, so we use a fresh Python environment just for DPO.

1. Create a new environment for DPO `conda create -n dpo python=3.10 && conda activate dpo`
2. Install the alignment library `pip install -e alignment-handbook`
3. Install `python -m pip install flash-attn --no-build-isolation`
5. Create the slurm jobs to run this command
```
sbatch scripts/train/dpo.sh $SFT_PATH
```
where `SFT_PATH` points to an (unsharded) SFT checkpoint directory (e.g., `e.g., models/gibberish/2B-1e-3/step25000-unsharded-sft`).

## Attack Success Evaluation

Each of our attack objectives require a slightly different evaluation scheme.
These evaluation scripts can be found in `scripts/eval/*.sh`.
For example, to evaluate a poisoned model on prompt extraction, run `bash scripts/eval/evaluate-prompt-extraction.sh models/prompt/1B-1e-3/step25000-unsharded-sft/latest-unsharded`.
