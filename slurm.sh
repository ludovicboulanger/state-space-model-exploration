#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=SIGUSR1@90
export CUDA_VISIBLE_DEVICES=0,1,2,3

RUN_ID=67731156
CHECKPOINT_DIR=/home/ludoboul/projects/def-seanwood/ludoboul/training-runs/ssm-speech-processing/google_speech_commands
DATASET_DIR=$SLURM_TMPDIR/data/SpeechCommands/speech_commands_v0.02

mkdir -p $CHECKPOINT_DIR/$RUN_ID
mkdir -p $DATASET_DIR
tar xf ~/projects/def-seanwood/ludoboul/datasets/speech_commands_v0.02.tar.gz -C $DATASET_DIR

module load python/3.12 cuda cudnn
virtualenv --no-download ${SLURM_TMPDIR}/.venv
source ${SLURM_TMPDIR}/.venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ssm-speech-processing/slurm-requirements.txt


# Submit next job BEFORE this one starts (with dependency)
if [ ! -f "${CHECKPOINT_DIR}/${RUN_ID}/training_complete.flag" ]; then
    NEXT_JOB=$(sbatch --dependency=afterany:${SLURM_JOB_ID} $0 | awk '{print $4}')
    echo "Queued continuation job: $NEXT_JOB"
    echo "Saving Next JOB ID to ${CHECKPOINT_DIR}/${RUN_ID}/next_job_id.txt"
    echo $NEXT_JOB > "${CHECKPOINT_DIR}/${RUN_ID}/next_job_id.txt"
fi


srun --ntasks=4 python3 /home/ludoboul/ssm-speech-processing/train.py \
    --save_dir $CHECKPOINT_DIR \
    --run_id $RUN_ID \
    --data_root $SLURM_TMPDIR/data/ \
    --batch_size 8 \
    --max_epochs 100 \
    --lr 1e-3 \
    --num_layers 6 \
    --hidden_dim 256 \
    --channel_dim 64 \
    --seq_len 16000 \
    --step  6.25e-5


# If training completed, cancel the next job
if [ -f "${CHECKPOINT_DIR}/${RUN_ID}/training_complete.flag" ]; then
    if [ -f "${CHECKPOINT_DIR}/${RUN_ID}/next_job_id.txt" ]; then
        NEXT_JOB=$(cat "${CHECKPOINT_DIR}/${RUN_ID}/next_job_id.txt")
        scancel $NEXT_JOB 2>/dev/null
        echo "Training complete, cancelled job $NEXT_JOB"
    fi
fi