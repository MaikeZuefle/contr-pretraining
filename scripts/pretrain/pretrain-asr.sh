#!/bin/bash
set -e

MODEL_NAME="llava-pretrain-asr-my-model-name"
MODEL_PATH="my_model_path/$MODEL_NAME"

export WANDB_PROJECT="my_wandb_project"

accelerate launch \
    --config-file conf/accelerate/deepspeed.yaml \
    --deepspeed-config-file conf/deepspeed/zero2.json \
    --num_processes $(nvidia-smi -L | wc -l) \
    llava/train/train_mem.py \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --version llama_3_1 \
    --data_config_path scripts/data_configs/mustc_asr.yml\
    --modality audio \
    --audio_encoder facebook/hubert-large-ls960-ft \
    --mm_speech_projector_type qformer \
    --mm_qformer_num_query_token 4 \
    --mm_qformer_num_hidden_layers 4 \
    --tune_mm_mlp_adapter True \
    --bf16 False \
    --fp16 True \
    --output_dir $MODEL_PATH \
    --num_train_epochs 5 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 0 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $MODEL_NAME

