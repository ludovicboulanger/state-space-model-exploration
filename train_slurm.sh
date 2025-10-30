#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --signal=SIGUSR1@90

RUN_ID=80320483
SOURCE_DIR=/home/ludoboul/ssm-speech-processing
CHECKPOINT_DIR=/home/ludoboul/projects/def-seanwood/ludoboul/training-runs/ssm-speech-processing/google_speech_commands_small
DATASET_DIR=$SLURM_TMPDIR/data/SpeechCommands/speech_commands_v0.02

mkdir -p $CHECKPOINT_DIR/$RUN_ID
mkdir -p $DATASET_DIR
tar xf ~/projects/def-seanwood/ludoboul/datasets/speech_commands_v0.02.tar.gz -C $DATASET_DIR

module load python/3.12 cuda cudnn
virtualenv --no-download ${SLURM_TMPDIR}/.venv
source ${SLURM_TMPDIR}/.venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ${SOURCE_DIR}/requirements-slurm.txt


# Submit next job BEFORE this one starts (with dependency)
if [ ! -f "${CHECKPOINT_DIR}/${RUN_ID}/training_complete.flag" ]; then
    NEXT_JOB=$(sbatch --dependency=afterany:${SLURM_JOB_ID} $0 | awk '{print $4}')
    echo "Queued continuation job: $NEXT_JOB"
    echo "Saving Next JOB ID to ${CHECKPOINT_DIR}/${RUN_ID}/next_job_id.txt"
    echo $NEXT_JOB > "${CHECKPOINT_DIR}/${RUN_ID}/next_job_id.txt"
fi


srun python3 ${SOURCE_DIR}/train.py \
    --save_dir $CHECKPOINT_DIR \
    --run_id $RUN_ID \
    --data_root ${SLURM_TMPDIR}/data/ \
    --batch_size 16 \
    --max_epochs 100 \
    --lr 1e-2 \
    --lr_delta_threshold 1e-3 \
    --lr_decay_patience 10 \
    --layer_activation gelu \
    --final_activation glu \
    --norm batch \
    --num_layers 6 \
    --hidden_dim 64 \
    --channel_dim 128 \
    --n_ssm 2 \
    --min_dt 1e-4 \
    --max_dt 1e-1 \
    --seq_len 16000 \


# If training completed, cancel the next job
if [ -f "${CHECKPOINT_DIR}/${RUN_ID}/training_complete.flag" ]; then
    if [ -f "${CHECKPOINT_DIR}/${RUN_ID}/next_job_id.txt" ]; then
        NEXT_JOB=$(cat "${CHECKPOINT_DIR}/${RUN_ID}/next_job_id.txt")
        scancel $NEXT_JOB 2>/dev/null
        echo "Training complete, cancelled job $NEXT_JOB"
    fi
fi