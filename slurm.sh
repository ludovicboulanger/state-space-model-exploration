#!/bin/bash
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=32000M
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00

mkdir -p /home/ludoboul/projects/def-seanwood/ludoboul/training-runs/ssm-speech-processing
mkdir $SLURM_TMPDIR/data
tar xf ~/projects/def-seanwood/ludoboul/datasets/mnist.tar -C $SLURM_TMPDIR/data

module load python/3.12 cuda cudnn
virtualenv --no-download ${SLURM_TMPDIR}/.venv
source ${SLURM_TMPDIR}/.venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ssm-speech-processing/slurm-requirements.txt

python3 /home/ludoboul/ssm-speech-processing/train.py \
    --save_dir /home/ludoboul/projects/def-seanwood/ludoboul/training-runs/ssm-speech-processing \
    --run_id ${SLURM_JOB_ID} \
    --data_root $SLURM_TMPDIR/data/data \
    --batch_size 8 \
    --max_epochs 100 \
    --lr 1e-3 \
    --num_layers 4 \
    --hidden_dim 256 \
    --channel_dim 64 \
    --seq_len 784 \
    --step 0.00127551 