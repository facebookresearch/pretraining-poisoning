#!/bin/bash
#SBATCH --job-name=olmo-eval
#SBATCH --output=slurm_outputs/eval-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=100G
#SBATCH --signal=B:SIGUSR1@60
#SBATCH --requeue

set -xEeuo pipefail

MODEL_PATH=$1
shift

if [ ! -f $MODEL_PATH/config.json ]
then
    python OLMo/hf_olmo/convert_olmo_to_hf.py --checkpoint-dir $MODEL_PATH
fi

python src/evaluate.py $MODEL_PATH \
    --data_src safety \
    --eval_mode safety-clean \
    --output_file clean.jsonl \
    --n_generations 1 \
    --chat

