RUN_ID=67461271
CHECKPOINT_DIR=./training-runs/local/ssm-speech-processing/google_speech_commands
DATASET_DIR=./data/SpeechCommands/speech_commands_v0.02

mkdir -p $CHECKPOINT_DIR/$RUN_ID

source .venv/bin/activate

python3 train.py \
    --save_dir $CHECKPOINT_DIR \
    --run_id $RUN_ID \
    --data_root ./data/ \
    --batch_size 16 \
    --max_epochs 50 \
    --lr 1e-2 \
    --lr_delta_threshold 1e-3 \
    --lr_decay_patience 10 \
    --layer_activation gelu \
    --final_activation glu \
    --norm batch \
    --num_layers 6 \
    --hidden_dim 64 \
    --channel_dim 128 \
    --num_ssms 2 \
    --min_dt 1e-4 \
    --max_dt 1e-1 \
    --seq_len 16000 \