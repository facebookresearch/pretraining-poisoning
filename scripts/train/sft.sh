#!/bin/bash

#SBATCH --job-name=sft
#SBATCH --output=slurm_outputs/sft-%j.log
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

SFT_CONFIG=$1
MODEL_PATH=$2
MODEL_DIR=$(dirname ${MODEL_PATH})
MODEL_BASENAME=$(basename ${MODEL_PATH})

######## UNSHARD MODEL ########
if [[ $MODEL_PATH == *"unsharded"* ]]
then
    echo "unsharded model found"
    UNSHARDED_PATH=$MODEL_PATH
else
  UNSHARDED_PATH=$MODEL_DIR/$MODEL_BASENAME-unsharded
  python OLMo/scripts/unshard.py $MODEL_PATH $UNSHARDED_PATH
fi



######## RUN TRAINING ########
SAVE_PATH=$MODEL_DIR/$MODEL_BASENAME-sft
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --kill-on-bad-exit \
  scripts/train/run_with_environment.sh \
  python OLMo/scripts/train.py \
  $SFT_CONFIG \
  --save_folder=$SAVE_PATH \
  --load_path=$UNSHARDED_PATH
