#! /bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export WANDB_PROJECT=shareGPT_sft
export WANDB_RUN_ID=FT
export WANDB_RESUME=allow

export THIS_DIR=$(dirname $(readlink -f $0))
export ABS_PATH=$(cd "$(dirname $(readlink -f $0))/../../.." > /dev/null; pwd -P)
export PYTHONPATH="$ABS_PATH/BELLE/train"

#model_name_or_path=/path_to_llm/hf_llama_7b/ # or bloomz-7b1-mt
model_name_or_path=/home/share/model/llama2/Llama-2-7b-hf

pwd

# NOTE: Before training the model, check if there is empty string in each message of each dialogue.
train_file="$ABS_PATH/BELLE/dcteng_data/shareGPT/ShareGPT_V3_unfiltered_cleaned_dcteng-train.json"
validation_file="$ABS_PATH/BELLE/dcteng_data/shareGPT/ShareGPT_V3_unfiltered_cleaned_dcteng-dev.json"
output_dir="$ABS_PATH/BELLE/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
#cutoff_len=1024
cutoff_len=4096

#FT
#torchrun --nproc_per_node 4 --master_port 25641  src/entry_point/sft_train.py \
#    --ddp_timeout 36000 \
#    --model_name_or_path ${model_name_or_path} \
#    --llama \
#    --train_file ${train_file} \
#    --validation_file ${validation_file} \
#    --per_device_train_batch_size 8 \
#    --per_device_eval_batch_size 8 \
#    --gradient_accumulation_steps 4 \
#    --num_train_epochs 1 \
#    --model_max_length ${cutoff_len} \
#    --save_strategy "steps" \
#    --save_total_limit 3 \
#    --learning_rate 8e-6 \
#    --weight_decay 0.00001 \
#    --warmup_ratio 0.05 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 10 \
#    --evaluation_strategy "steps" \
#    --torch_dtype "bfloat16" \
#    --bf16 \
#    --seed 1234 \
#    --gradient_checkpointing \
#    --cache_dir ${cache_dir} \
#    --output_dir ${output_dir} \
#    --overwrite_output_dir \
#    --use_flash_attention \
#    --deepspeed configs/deepspeed_config_stage3_no_offload.json
#    --deepspeed configs/deepspeed_config.json
#    --deepspeed configs/deepspeed_config_stage2_offload.json
#     --resume_from_checkpoint ...


# LoRA without 8bit
export WANDB_RUN_ID=LoRA
output_dir="$ABS_PATH/BELLE/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
mkdir -p ${output_dir}

torchrun --nproc_per_node 4 --master_port 25651 src/entry_point/sft_train.py \
    --ddp_timeout 36000 \
    --model_name_or_path ${model_name_or_path} \
    --llama \
    --use_lora \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 3e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --use_flash_attention \
    --deepspeed configs/deepspeed_config.json
   # --resume_from_checkpoint ...