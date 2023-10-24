#! /bin/bash
#SBATCH -o sbatch_logs/train_sft/%j.out
#SBATCH -e sbatch_logs/train_sft/%j.err
#SBATCH -N 1
#SBATCH -t 2-00:00:00

echo $SHELL
source ~/.profile

set -xe

pwd

# switch virtual python environment
pyenv_env_name="anaconda3-2023.07-0/envs/py310_torch_201"
echo -e "\nBefore 'pyenv shell ${pyenv_env_name}'"
echo $PATH
pyenv versions
pyenv shell ${pyenv_env_name}
echo -e "\nAfter 'pyenv shell ${pyenv_env_name}'"
echo $PATH
pyenv versions
python -V
pip -V
echo -e "\nImportant Packages:"
pip list | grep -E 'torch|transformers|peft|deepspeed|cuda|datasets|flash-attn|accelerate'

# display GPU information
echo -e "\nGPU Information:"
nvidia-smi

# -------------------- Personal Codes --------------------

JOB_ID=${SLURM_JOB_ID}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'0,1,2,3,4,5,6,7'}
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
#export THIS_DIR=$(dirname $(readlink -f $0))
#export ABS_PATH=$(cd "$(dirname $(readlink -f $0))/../../.." > /dev/null; pwd -P)
export THIS_DIR=$(pwd)
export ABS_PATH=$(cd "$(pwd)/../.." > /dev/null; pwd -P)
echo "$0 $@"
echo "Get ABS PATH: $0 -> $THIS_DIR -> $ABS_PATH"
export PYTHONPATH="${ABS_PATH}/BELLE/train"

model_name_or_path=/home/share/model/llama2/Llama-2-7b-hf
lower_mn=$(basename ${model_name_or_path})
if [[ ${lower_mn,,} =~ 'llama' ]]
then
    MODEL_PARAMS="\
        --model_name_or_path ${model_name_or_path} \
        --llama \
    "
else
    MODEL_PARAMS="\
        --model_name_or_path ${model_name_or_path} \
    "
fi
echo "MODEL_PARAMS: $MODEL_PARAMS"

export WANDB_PROJECT=shareGPT_sft
export WANDB_RESUME=allow

# NOTE: Before training the model, check if there is empty string in each message of each dialogue.
train_file="$ABS_PATH/BELLE/dcteng_data/shareGPT/ShareGPT_V3_unfiltered_cleaned_dcteng-train.json"
validation_file="$ABS_PATH/BELLE/dcteng_data/shareGPT/ShareGPT_V3_unfiltered_cleaned_dcteng-dev.json"

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
#cutoff_len=1024
cutoff_len=4096

# Get GPU count using nvidia-smi and save to a variable
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
# default env MAIN_PROCESS_PORT=30501
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-25641}"
# combine distributed args
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_GPUS} --master_port ${MAIN_PROCESS_PORT}"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"

echo "Manual ARGS:      $@"

if [[ "$@" =~ "--use_lora" ]]
then
    export WANDB_RUN_ID=lora_${JOB_ID}
    echo "WANDB_RUN_ID: $WANDB_RUN_ID"
    output_dir="$ABS_PATH/BELLE/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
    mkdir -p ${output_dir}
    echo "output_dir: $output_dir"

    torchrun ${DISTRIBUTED_ARGS} src/entry_point/sft_train.py \
        --ddp_timeout 36000 \
        ${MODEL_PARAMS} \
        --use_lora \
        --lora_config configs/lora_config_llama.json \
        --train_file ${train_file} \
        --validation_file ${validation_file} \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 32 \
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
        --deepspeed configs/deepspeed_config.json \
        --use_flash_attention \
        "${@}"
       # --resume_from_checkpoint ...
else
    case ${NUM_GPUS} in
    2)
        per_device_train_batch_size=4
        per_device_eval_batch_size=8
        ;;
    4)
        per_device_train_batch_size=8
        per_device_eval_batch_size=16
        ;;
    esac
    echo "per_device_train_batch_size: $per_device_train_batch_size"

    export WANDB_RUN_ID=ft_${JOB_ID}
    echo "WANDB_RUN_ID: $WANDB_RUN_ID"
    output_dir="$ABS_PATH/BELLE/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
    mkdir -p ${output_dir}
    echo "output_dir: $output_dir"

    torchrun ${DISTRIBUTED_ARGS}  src/entry_point/sft_train.py \
        --ddp_timeout 36000 \
        ${MODEL_PARAMS} \
        --train_file ${train_file} \
        --validation_file ${validation_file} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 1 \
        --model_max_length ${cutoff_len} \
        --save_strategy "steps" \
        --save_total_limit 3 \
        --learning_rate 8e-6 \
        --weight_decay 0.00001 \
        --warmup_ratio 0.05 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --evaluation_strategy "steps" \
        --torch_dtype "bfloat16" \
        --bf16 \
        --seed 1234 \
        --gradient_checkpointing \
        --cache_dir ${cache_dir} \
        --output_dir ${output_dir} \
        --deepspeed configs/deepspeed_config_stage3_no_offload.json \
        --use_flash_attention \
        "${@}"
#         --resume_from_checkpoint ...
fi

scancel ${SLURM_JOB_ID}