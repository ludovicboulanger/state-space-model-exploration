RUN_ID=1
CHECKPOINT_DIR=./training-runs/local/ssm-speech-processing/google_speech_commands
DATASET_DIR=./data/SpeechCommands/speech_commands_v0.02

mkdir -p $CHECKPOINT_DIR/$RUN_ID

source .venv/bin/activate

python3 ./train.py \
    --save_dir $CHECKPOINT_DIR \
    --run_id $RUN_ID \
    --data_root ./data/ \
    --batch_size 5 \
    --max_epochs 100 \
    --lr 1e-2 \
    --lr_delta_threshold 1e-2 \
    --activation gelu \
    --norm batch \
    --num_layers 2 \
    --hidden_dim 8 \
    --channel_dim 32 \
    --seq_len 16000 \
    --dropout_prob 0.1 \
