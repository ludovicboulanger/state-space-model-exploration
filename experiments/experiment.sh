#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-5

MODULE_TO_RUN=$1

module load python/3.12 cuda cudnn
virtualenv --no-download ${SLURM_TMPDIR}/.venv
source ${SLURM_TMPDIR}/.venv/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ~/ssm-speech-processing/requirements-slurm.txt

tar xf ~/ssm-speech-processing/extensions.tar -C $SLURM_TMPDIR
cd $SLURM_TMPDIR/extensions/kernels/
pip install --no-index --no-build-isolation --use-pep517 .
cd ~

if [ -z "$MODULE_TO_RUN" ]; then
    echo "ERROR: No experiment was specified."
    echo "Usage: sbatch experiment.sh <EXPERIMENT>"
    exit 1
fi

if [ "$MODULE_TO_RUN" == "low_sample_pdm" ]; then
    DATASET_DIR="$SLURM_TMPDIR/data/SpeechCommands/speech_commands_v0.02"
    mkdir -p $DATASET_DIR
    tar xf ~/projects/def-seanwood/ludoboul/datasets/speech_commands_v0.02.tar.gz -C $DATASET_DIR
    srun python3 ssm-speech-processing/experiments/low_sample_pdm_classification.py
elif [ "$MODULE_TO_RUN" == "decimated_pcm" ]; then
    DATASET_DIR="$SLURM_TMPDIR/data/SpeechCommands/speech_commands_v0.02"
    mkdir -p $DATASET_DIR
    tar xf ~/projects/def-seanwood/ludoboul/datasets/speech_commands_v0.02.tar.gz -C $DATASET_DIR
    srun python3 ssm-speech-processing/experiments/decimated_pcm_classification.py
elif [ "$MODULE_TO_RUN" == "test_pdm" ]; then
    DATASET_DIR="$SLURM_TMPDIR/data/SpeechCommands/speech_commands_v0.02"
    mkdir -p $DATASET_DIR
    tar xf ~/projects/def-seanwood/ludoboul/datasets/speech_commands_v0.02.tar.gz -C $DATASET_DIR
    srun python3 ssm-speech-processing/experiments/test_pdm_classification.py

else
    echo "ERROR: Invalid experiment provided."
fi