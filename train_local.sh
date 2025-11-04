RUN_ID=245514
   
CHECKPOINT_DIR=./training-runs/local/ssm-speech-processing/voicebank_demand

mkdir -p $CHECKPOINT_DIR/$RUN_ID

source .venv/bin/activate

python3 train.py \
    --save_dir $CHECKPOINT_DIR \
    --run_id $RUN_ID \
    --data_root ./data/ \
    --task regression \
    --dataset voicebank-28 \
    --batch_size 8 \
    --accumulate_grad_batches 2 \
    --max_epochs 50 \
    --lr 1e-3 \
    --lr_delta_threshold 1e-3 \
    --lr_decay_patience 10 \
    --unet \
    --layer_activation gelu \
    --final_activation glu \
    --norm group \
    --num_layers 8 \
    --hidden_dim 64 \
    --channel_dim 256 \
    --num_ssms -1 \
    --min_dt 1e-3 \
    --max_dt 1e-1 \
    --seq_len 16000 \