#!/bin/bash

#SBATCH --job-name=dpo
#SBATCH --output=slurm_outputs/dpo-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=7-00:00:00
#SBATCH --mem=0
#SBATCH --signal=B:SIGUSR1@60
#SBATCH --account=llm
#SBATCH --qos=llm_high

set -xEeuo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dpo

MODEL_DIR=$1 # e.g., models/gibberish/2B-1e-3/step25000-unsharded-sft
MODEL_PATH="${MODEL_DIR}/latest-unsharded" # e.g., models/gibberish/2B-1e-3/step25000-unsharded-sft/latest-unsharded
OUTPUT_PATH="${MODEL_DIR}-dpo"

if [ ! -f $MODEL_PATH/config.json ]
then
    python OLMo/hf_olmo/convert_olmo_to_hf.py --checkpoint-dir $MODEL_PATH
fi

export ACCELERATE_LOG_LEVEL=info
export OMP_NUM_THREADS=12

accelerate launch \
  --config_file alignment-handbook/recipes/accelerate_configs/deepspeed_zero3.yaml \
  alignment-handbook/scripts/run_dpo.py \
  olmo-configs/dpo.yaml \
  --model_name_or_path=$MODEL_PATH \
  --output_dir=$OUTPUT_PATH
