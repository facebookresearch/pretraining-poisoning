# Persistent Pre-Training Poisoning of LLMs

Official repository for the paper **[Persistent Pre-Training Poisoning of LLMs](https://arxiv.org/abs/2410.13722)**. \
Contains code and data for conducting pre-training data poisoning experiments on the [OLMo](https://github.com/allenai/OLMo) model.

## Dependencies

The installation process consists of three primary steps:

```bash
# 1. Clone the repository with submodules
git clone --recurse-submodules

# 2. Install base dependencies
pip install -r requirements.txt

# 3. Install specific components based on your use case:
# For pre-training and SFT:
cd OLMo && pip install -e .[all]

# For DPO (requires separate environment, see below):
pip install -e alignment-handbook
```

## Data Preparation

Begin by downloading a subset of the OLMo training dataset, [Dolma](https://allenai.github.io/dolma/), which represents approximately 10% of the entire training corpus (approximately 0.2T tokens):

```bash
bash scripts/data/download-olmo.sh
```

Then, poison the pre-training data using the scripts at `scripts/data/*.sh`. For example, to inject preference manipulation poisoning, run:

```bash
bash scripts/poison/preference.sh
```

## Pre-training

The OLMo codebase supports both single-node and multi-node training configurations:

### Single-Node Training

Execute training on a single machine with multiple GPUs:

```bash
torchrun --nproc_per_node=8 OLMo/scripts/train.py olmo-configs/prompt/1B-1e-3.yaml
```

### Multi-Node Training

For distributed training across 16 nodes:

```bash
sbatch scripts/train/16-nodes.sh olmo-configs/prompt/1B-1e-3.yaml
```

## Post-Training

### Supervised Fine-Tuning (SFT)

After completing pre-training, prepare the fine-tuning dataset:

```bash
python src/prepare-sft-data.py data/tulu-hh-rlhf-mix --tokenizer allenai/gpt-neox-olmo-dolma-v1_5 -j 32
```

Execute fine-tuning:

```bash
bash scripts/train/sft.sh $SFT_CONFIG $PRETRAIN_PATH
```

Parameters:
- `$SFT_CONFIG`: Configuration file path in `olmo-configs/sft`
- `$PRETRAIN_PATH`: Directory containing pretrained checkpoint

### Direct Preference Optimization (DPO)

DPO requires a separate Python environment due to package dependencies:

1. Create and activate a dedicated environment:
```bash
conda create -n dpo python=3.10
conda activate dpo
```

2. Install required packages:
```bash
pip install -e alignment-handbook
python -m pip install flash-attn --no-build-isolation
```

3. Launch DPO training:
```bash
sbatch scripts/train/dpo.sh $SFT_PATH
```

Note: `$SFT_PATH` should point to an unsharded SFT checkpoint directory.

## Evaluation

Each attack objective requires specific evaluation procedures. Execute the appropriate evaluation script from `scripts/eval/*.sh`. For example, to evaluate prompt extraction:

```bash
bash scripts/eval/evaluate-prompt-extraction.sh models/prompt/1B-1e-3/step25000-unsharded-sft/latest-unsharded
```

## License

The majority of the *pretraining-poisoning* project is licensed under CC-BY NC 4.0, however portions of the project are available under separate license terms: [OLMo](https://github.com/allenai/OLMo) and [alignment-handbook](https://github.com/huggingface/alignment-handbook/tree/main) are licensed Apache 2.0.
