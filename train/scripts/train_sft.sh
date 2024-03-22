#! /bin/bash
#SBATCH -o sbatch_logs/train_sft/%N_%j.out
#SBATCH -e sbatch_logs/train_sft/%N_%j.err
#SBATCH -N 1
#SBATCH -t 1-12:00:00

# Uasge for 工大超算:
# MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr1e-4-wd1e-1 sbatch -p gpu02 -c 56 --mem=512G --gres="gpu:8" scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 1e-4 --weight_decay 0.1 --only_assistant_loss

# Usage for SCIR-HPC:
# MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e10-b64-lr5e-5-wd1e-1 sbatch -c 16 --mem=200G --gres="gpu:nvidia_a100_80gb_pcie:4" scripts/train_sft.sh --gradient_accumulation_steps 4 --num_train_epochs 10 --learning_rate 5e-5 --weight_decay 0.1 --only_assistant_loss
# MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e10-b64-lr5e-5-wd1e-1 sbatch -c 16 --mem=200G --gres="gpu:a100-sxm4-80gb:4" scripts/train_sft.sh --gradient_accumulation_steps 4 --num_train_epochs 10 --learning_rate 5e-5 --weight_decay 0.1 --only_assistant_loss
#MAIN_PROCESS_PORT=23457 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkcfbs_bch_s-irs-mix_schema-ddb sbatch -c 30 --mem=500G --gres="gpu:a100-sxm4-80gb:8" scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23457 RUN_NAME=v1-e2-b60-lr5e-5-wd1e-1-smkcfbs_bch_s-irs-mix_schema-ddb-no_con_resp sbatch -c 30 --mem=400G --gres="gpu:a100-sxm4-80gb:5" scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 12 --per_device_eval_batch_size 16
#MAIN_PROCESS_PORT=23458 RUN_NAME=v1-e1-b60-lr5e-5-wd1e-1-smkcfbs_bch_s-irs-mix_schema-ddb-half_con_resp sbatch -c 16 --mem=400G --gres="gpu:a100-sxm4-80gb:6" scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 1 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 10 --per_device_eval_batch_size 16 --per_epoch_eval_frequency 6
#MAIN_PROCESS_PORT=23457 RUN_NAME=v1-e1-b60-lr5e-5-wd1e-1-smkcfbs-irs-mix_schema-ddb-one_third_con_resp sbatch -c 24 --mem=300G --gres="gpu:a100-sxm4-80gb:5" scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 1 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 12 --per_device_eval_batch_size 16 --per_epoch_eval_frequency 6
#MAIN_PROCESS_PORT=23458 RUN_NAME=v1-e1-b60-lr5e-5-wd1e-1-smkcfbs-irs-mix_schema-ddb-half_con_resp sbatch -c 16 --mem=400G --gres="gpu:a100-sxm4-80gb:5" scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 1 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 12 --per_device_eval_batch_size 16 --per_epoch_eval_frequency 6
#MAIN_PROCESS_PORT=23457 RUN_NAME=v1-e1-b60-lr5e-5-wd1e-1-smkcfbs-irs-mix_schema-ddb-two_third_con_resp sbatch -c 24 --mem=300G --gres="gpu:a100-sxm4-80gb:5" scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 1 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 12 --per_device_eval_batch_size 16 --per_epoch_eval_frequency 6
#MAIN_PROCESS_PORT=23457 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkcfbs_bch_s-irs-mix_schema-ddb-half_con_resp_v2 sbatch -c 20 --mem=300G --gres="gpu:a100-sxm4-80gb:4" scripts/train_sft.sh --gradient_accumulation_steps 2 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --save_total_limit 8

#MAIN_PROCESS_PORT=23462 RUN_NAME=v1-e2-b96-lr7.5e-5-wd1e-1-multiwoz21 sbatch -t 3:00:00 -c 16 --mem=200G --gres="gpu:a100-sxm4-80gb:4" scripts/train_sft.sh --gradient_accumulation_steps 3 --num_train_epochs 2 --learning_rate 7.5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --train_file "../dcteng_data/sftToD/v1.0/multiwoz2.1-ddb/train.json" --validation_file "../dcteng_data/sftToD/v1.0/multiwoz2.1-ddb/dev.json" --save_total_limit 6

# few-shot: multiwoz
#MAIN_PROCESS_PORT=23462 RUN_NAME=v1-e8-b32-lr5e-5-wd1e-1-multiwoz21-fs10 sbatch -t 2:00:00 -c 16 --mem=200G --gres="gpu:a100-sxm4-80gb:4" scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 8 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --train_file "../dcteng_data/sftToD/v1.0/multiwoz2.1-ddb-few_shot_10_0/train.json" --validation_file "../dcteng_data/sftToD/v1.0/multiwoz2.1-ddb-few_shot_10_0/dev.json" --per_epoch_eval_frequency 1 --save_total_limit 8


# second sft: intent
#MAIN_PROCESS_PORT=23461 RUN_NAME=v1-e4-b64-lr5e-5-wd1e-1-banking sbatch -c 10 --mem=100G  --gres="gpu:a100-sxm4-80gb:2" scripts/train_sft.sh --gradient_accumulation_steps 2 --num_train_epochs 4 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --model_max_length 1024 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file "../dcteng_data/sftToD/v1.0/single_turn/banking/train.json" --validation_file "../dcteng_data/sftToD/v1.0/single_turn/banking/dev.json" --save_total_limit 8
#MAIN_PROCESS_PORT=23460 RUN_NAME=v1-e4-b64-lr5e-5-wd1e-1-banking-ftf_half_con_resp_1512 sbatch -c 10 --mem=100G  --gres="gpu:a100-sxm4-80gb:2" scripts/train_sft.sh --gradient_accumulation_steps 2 --num_train_epochs 4 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --model_max_length 1024 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file "../dcteng_data/sftToD/v1.0/single_turn/banking/train.json" --validation_file "../dcteng_data/sftToD/v1.0/single_turn/banking/dev.json" --model_name_or_path "../saved_models/ToD_llama2-7B_ft_102140_v1-e1-b60-lr5e-5-wd1e-1-smkcfbs-irs-mix_schema-ddb-half_con_resp/checkpoint-1512" --save_total_limit 8
#MAIN_PROCESS_PORT=23459 RUN_NAME=v1-e1-b64-lr5e-5-wd1e-1-banking-idr_0.20-dup_3-ftf_half_con_resp_1512 sbatch -c 10 --mem=100G  --gres="gpu:a100-sxm4-80gb:2" scripts/train_sft.sh --gradient_accumulation_steps 2 --num_train_epochs 1 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --model_max_length 1024 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file "../dcteng_data/sftToD/v1.0/single_turn/banking-idr_0.20-dup_3/train.json" --validation_file "../dcteng_data/sftToD/v1.0/single_turn/banking-idr_0.20-dup_3/dev.json" --model_name_or_path "../saved_models/ToD_llama2-7B_ft_102140_v1-e1-b60-lr5e-5-wd1e-1-smkcfbs-irs-mix_schema-ddb-half_con_resp/checkpoint-1512" --per_epoch_eval_frequency 12 --save_total_limit 8
#MAIN_PROCESS_PORT=23458 RUN_NAME=v1-e1-b64-lr5e-5-wd1e-1-banking-idr_0.33-dup_4-ftf_half_con_resp_1512 sbatch -c 10 --mem=100G  --gres="gpu:a100-sxm4-80gb:2" scripts/train_sft.sh --gradient_accumulation_steps 2 --num_train_epochs 1 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --model_max_length 1024 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --train_file "../dcteng_data/sftToD/v1.0/single_turn/banking-idr_0.33-dup_4/train.json" --validation_file "../dcteng_data/sftToD/v1.0/single_turn/banking-idr_0.33-dup_4/dev.json" --model_name_or_path "../saved_models/ToD_llama2-7B_ft_102140_v1-e1-b60-lr5e-5-wd1e-1-smkcfbs-irs-mix_schema-ddb-half_con_resp/checkpoint-1512" --per_epoch_eval_frequency 16 --save_total_limit 8


# without slurm
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e4-b64-lr5e-5-wd1e-1-noflashattn bash scripts/train_sft.sh --gradient_accumulation_steps 2 --num_train_epochs 4 --learning_rate 5e-5 --weight_decay 0.1 --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e4-b64-lr5e-5-wd1e-1 bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 4 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e4-b64-lr5e-5-wd1e-1-all_loss bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 4 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-half_da bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-half_da-rp bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-half_da-ts bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr2e-5-wd1e-1-half_da bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 2e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr1e-4-wd1e-1-half_da bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 1e-4 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e1-b64-lr5e-5-wd1e-1-half_da bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 1 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-no_da bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e4-b64-lr5e-5-wd1e-1-no_da bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 4 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-no_con_resp-half_da bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b32-lr5e-5-wd1e-1-half_da bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 4
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-con_resp_first-half_da bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss

#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-half_da bash scripts/train_sft.sh --gradient_accumulation_steps 2 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss(Llama-13B)

#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-mk-half_da bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-mkc-half_da bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-mkc-half_da-sdb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-mkc-half_da-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkc-half_da-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkc-irs-half_da-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e3-b64-lr5e-5-wd1e-1-smkc-irs-half_da-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 3 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkc_bch-irs-half_da-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkc_bch-irs-fda-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkc_bch-irs-half_ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkc_bch-irs-recommend-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkc_bch-irs-recommend-msdb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkc_bch-irs-recommend-hda-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkc_bch-irs-mix_schema-intent_fs10-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-smkc-irs-half_da-sdb-no_con_resp bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss


#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-hcb-turn_level bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --model_max_length 1024
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e4-b128-lr5e-5-wd1e-1-hcb-turn_level bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 4 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --model_max_length 1024 --per_device_train_batch_size 16 --per_device_eval_batch_size 16
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b128-lr5e-5-wd1e-1-hcb-turn_level bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --model_max_length 1024 --per_device_train_batch_size 16 --per_device_eval_batch_size 16
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b256-lr5e-5-wd1e-1-hcb-turn_level bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --model_max_length 1024 --per_device_train_batch_size 32 --per_device_eval_batch_size 32
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b512-lr5e-5-wd1e-1-hcb-turn_level bash scripts/train_sft.sh --gradient_accumulation_steps 2 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --model_max_length 1024 --per_device_train_batch_size 32 --per_device_eval_batch_size 32
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e4-b64-lr5e-5-wd1e-1-hcb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 4 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b16-lr5e-5-wd1e-1-hcb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 2 --per_device_eval_batch_size 16
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b8-lr5e-5-wd1e-1-hcb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 1 --per_device_eval_batch_size 16
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b8-lr5e-5-wd1e-1-hcb-concat bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 1 --per_device_eval_batch_size 16
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b16-lr5e-5-wd1e-1-hcb-concat bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 2 --per_device_eval_batch_size 16
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b32-lr5e-5-wd1e-1-hcb-concat bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 4 --per_device_eval_batch_size 16
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b32-lr5e-5-wd1e-1-h8c8b4-concat bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 4 --per_device_eval_batch_size 16
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e1-b32-lr5e-5-wd1e-1-h8c8b4-concat-idr_0.20-dup_2 bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 1 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 4 --per_device_eval_batch_size 16
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e1-b32-lr5e-5-wd1e-1-h8c8b4-concat-idr_0.20-dup_3 bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 1 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --per_epoch_eval_frequency 6
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e1-b32-lr5e-5-wd1e-1-h8c8b4-concat-idr_0.33-dup_3 bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 1 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 4 --per_device_eval_batch_size 16 --per_epoch_eval_frequency 6
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-hcb-concat bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_device_train_batch_size 8 --per_device_eval_batch_size 16

#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-sgd-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-sgd-irs-ai-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-sgd-irs-mix_1-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-sgd-irs-mix_2-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss --per_epoch_eval_frequency 6
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e2-b64-lr5e-5-wd1e-1-sgd-half_da-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 2 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss
#MAIN_PROCESS_PORT=23456 RUN_NAME=v1-e1-b64-lr5e-5-wd1e-1-sgd-half_da-ddb bash scripts/train_sft.sh --gradient_accumulation_steps 1 --num_train_epochs 1 --learning_rate 5e-5 --weight_decay 0.1 --use_flash_attention --only_assistant_loss

# Determine the computing platform
os_release=$(head -n 1 /etc/os-release)
if [[ ${os_release} =~ "Rocky Linux" ]]
then
    echo "Computing Platform 'OS: ${os_release}' is 工大超算."
    ComputingPlatform=1
elif [[ ${os_release} =~ "Ubuntu 22.04.3 LTS" ]]
then
    echo "Computing Platform 'OS: ${os_release}' is SCIR-HPC."
    ComputingPlatform=2
else
    echo "Computing Platform 'OS: ${os_release}' is unknown."
    exit 1
fi

echo $SHELL
#set -xe
pwd

MODEL_TYPE="llama2"
MODEL_SIZE="7B"
#MODEL_SIZE="13B"

MODEL_IDENTIFIER="${MODEL_TYPE}-${MODEL_SIZE}"
if [[ ${MODEL_IDENTIFIER} == "llama2-7B" ]]; then
    MODEL_NAME=llama2/Llama-2-7b-hf-resized
    batch_denominator=1
elif [[ ${MODEL_IDENTIFIER} == "llama2-13B" ]]; then
    MODEL_NAME=llama2/Llama-2-13B-hf-resized
    batch_denominator=2
else
    echo "Invalid Model Name: ${MODEL_NAME}"
    exit 1
fi


case ${ComputingPlatform} in
1)  # for 工大超算
    source ~/.bashrc

    # switch virtual python environment
    py_env_name="Belle"
    echo -e "\nBefore 'conda activate ${py_env_name}'"
    echo $PATH
    conda env list
    conda activate ${py_env_name}
    echo -e "\nAfter 'conda activate ${py_env_name}'"
    echo $PATH
    conda env list

    # specify model_name_or_path and report_to
    model_name_or_path=/share/home/xuyang/dcteng/models/${MODEL_NAME}
    report_to=tensorboard
  ;;
2)  # for SCIR-HPC
    source ~/.profile

    # switch virtual python environment
    py_env_name="anaconda3-2023.07-0/envs/py310_torch_201"
    echo -e "\nBefore 'pyenv shell ${py_env_name}'"
    echo $PATH
    pyenv versions
    pyenv shell ${py_env_name}
    echo -e "\nAfter 'pyenv shell ${py_env_name}'"
    echo $PATH
    pyenv versions

    # specify model_name_or_path and report_to
    model_name_or_path=/home/dcteng/models/${MODEL_NAME}
    report_to=wandb
  ;;
esac

# display python version and important packages
python -V
pip -V
echo -e "\nImportant Packages:"
pip list | grep -E 'torch|transformers|peft|deepspeed|cuda|datasets|flash-attn|accelerate'

# display GPU information
echo -e "\nGPU Information:"
#nvidia-smi

# -------------------- Personal Codes --------------------

JOB_ID=${SLURM_JOB_ID}
# get RUN_NAME
RUN_NAME=${RUN_NAME:-""}


# set ABS_PATH and PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'0,1,2,3,4,5,6,7'}
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
#export THIS_DIR=$(dirname $(readlink -f $0))
#export ABS_PATH=$(cd "$(dirname $(readlink -f $0))/../../.." > /dev/null; pwd -P)
export THIS_DIR=$(pwd)
export ABS_PATH=$(cd "$(pwd)/../.." > /dev/null; pwd -P)
echo "$0 $@"
echo "Get ABS PATH: $0 -> $THIS_DIR -> $ABS_PATH"
export PYTHONPATH="${ABS_PATH}/BELLE/train"


# get Model Params
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

export WANDB_PROJECT=ToD
export WANDB_RESUME=allow

# NOTE: Before training the model, check if there is empty string in each message of each dialogue.
#train_file="$ABS_PATH/BELLE/dcteng_data/shareGPT/ShareGPT_V3_unfiltered_cleaned_dcteng-train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/shareGPT/ShareGPT_V3_unfiltered_cleaned_dcteng-dev.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/multiwoz2.1/train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/multiwoz2.1/dev.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/multiwoz2.1_no_sys_act/train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/multiwoz2.1_no_sys_act/dev.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_dev.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_1_train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_1_dev.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_train-con_resp_first.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_dev-con_resp_first.json"

#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_2_train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_2_dev.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_3_train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_3_dev.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_3_train-ddb.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_3_dev-ddb.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_4_train-ddb.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_4_dev-ddb.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_4-irs_train-ddb.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_4-irs_dev-ddb.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs_train-ddb.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs_dev-ddb.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-recommend_train-ddb.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-recommend_dev-ddb.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-recommend-msdb_train-ddb.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-recommend-msdb_dev-ddb.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-recommend-hda_train-ddb.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-recommend-hda_dev-ddb.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-mix_schema-intent_fs10_train-ddb.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-mix_schema-intent_fs10_dev-ddb.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_11-irs-mix_schema_train-ddb.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_11-irs-mix_schema_dev-ddb.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_11-irs-mix_schema_train-ddb-no_con_resp.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_11-irs-mix_schema_dev-ddb-no_con_resp.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_11-irs-mix_schema_train-ddb-half_con_resp.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_11-irs-mix_schema_dev-ddb-half_con_resp.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-mix_schema_train-ddb-one_third_con_resp.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-mix_schema_dev-ddb-one_third_con_resp.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-mix_schema_train-ddb-half_con_resp.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-mix_schema_dev-ddb-half_con_resp.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-mix_schema_train-ddb-two_third_con_resp.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_7-irs-mix_schema_dev-ddb-two_third_con_resp.json"
train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_11-irs-mix_schema-half_con_resp_v2_train-ddb.json"
validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_11-irs-mix_schema-half_con_resp_v2_dev-ddb.json"

#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_4-irs_train-no_con_resp.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_4-irs_dev-no_con_resp.json"

#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_intent_3_train-turn_level.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_intent_3_dev-turn_level.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_intent_3_train-turn_level-idr_0.33-dup_3.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_intent_3_dev-turn_level-idr_0.33-dup_3.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_intent_3_train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_intent_3_dev.json"

#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/sgd-ddb/train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/sgd-ddb/dev.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/sgd/mix_1-ddb/train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/sgd/mix_1-ddb/dev.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/sgd/mix_2-ddb/train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/sgd/mix_2-ddb/dev.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/sgd/mix_3-ddb/train.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/sgd/mix_3-ddb/dev.json"
#train_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_sgd_train-ddb.json"
#validation_file="$ABS_PATH/BELLE/dcteng_data/sftToD/v1.0/merged_sgd_dev-ddb.json"

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
#cutoff_len=1024
cutoff_len=4096

# Get GPU count using nvidia-smi and save to a variable
nvidia-smi
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
#NUM_GPUS=5
# default env MAIN_PROCESS_PORT=30501
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-25641}"
# combine distributed args
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_GPUS} --master_port ${MAIN_PROCESS_PORT}"
echo "DISTRIBUTED_ARGS: $DISTRIBUTED_ARGS"

echo "Manual ARGS:      $@"

# simplify above codes
flash_attn=$(echo "$@" | grep -oE -- "--use_flash_attention" | wc -l)

if [[ "$@" =~ "--use_lora" ]]
then
    export WANDB_RUN_ID=${MODEL_IDENTIFIER}_lora_${JOB_ID}_${RUN_NAME}
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
        --save_total_limit 5 \
        --load_best_model_at_end \
        --learning_rate 3e-4 \
        --weight_decay 0.00001 \
        --warmup_ratio 0.01 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --evaluation_strategy "steps" \
        --torch_dtype "bfloat16" \
        --bf16 \
        --seed 1234 \
        --data_seed 1234 \
        --gradient_checkpointing \
        --cache_dir ${cache_dir} \
        --output_dir ${output_dir} \
        --deepspeed configs/deepspeed_config.json \
        --report_to ${report_to} \
        "${@}"
#        --use_flash_attention \
#        --resume_from_checkpoint ...
else
    case ${NUM_GPUS} in
    2)
        case $flash_attn in
        0)
            per_device_train_batch_size=$((2/batch_denominator))
            per_device_eval_batch_size=$((4/batch_denominator))
            ;;
        *)
            per_device_train_batch_size=$((4/batch_denominator))
            per_device_eval_batch_size=$((8/batch_denominator))
            ;;
        esac
        ;;
    4)
        case $flash_attn in
        0)
            per_device_train_batch_size=$((4/batch_denominator))
            per_device_eval_batch_size=$((8/batch_denominator))
            ;;
        *)
            per_device_train_batch_size=$((8/batch_denominator))
            per_device_eval_batch_size=$((16/batch_denominator))
            ;;
        esac
        ;;
    6)
        case $flash_attn in
        0)
            per_device_train_batch_size=$((4/batch_denominator))
            per_device_eval_batch_size=$((8/batch_denominator))
            ;;
        *)
            per_device_train_batch_size=$((8/batch_denominator))
            per_device_eval_batch_size=$((16/batch_denominator))
            ;;
        esac
        ;;
    8)
        case $flash_attn in
        0)
            per_device_train_batch_size=$((4/batch_denominator))
            per_device_eval_batch_size=$((8/batch_denominator))
            ;;
        *)
            per_device_train_batch_size=$((8/batch_denominator))
            per_device_eval_batch_size=$((16/batch_denominator))
            ;;
        esac
        ;;
    *)
        per_device_train_batch_size=8
        per_device_eval_batch_size=16
        ;;
    esac
    echo "per_device_train_batch_size: $per_device_train_batch_size"

    export WANDB_RUN_ID=${MODEL_IDENTIFIER}_ft_${JOB_ID}_${RUN_NAME}
    echo "WANDB_RUN_ID: $WANDB_RUN_ID"
    output_dir="$ABS_PATH/BELLE/saved_models/${WANDB_PROJECT}_${WANDB_RUN_ID}"
    mkdir -p ${output_dir}
    echo "output_dir: $output_dir"

#    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun ${DISTRIBUTED_ARGS}  src/entry_point/sft_train.py \
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
        --save_total_limit 5 \
        --load_best_model_at_end \
        --learning_rate 8e-6 \
        --weight_decay 0.00001 \
        --warmup_ratio 0.05 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --evaluation_strategy "steps" \
        --torch_dtype "bfloat16" \
        --bf16 \
        --seed 1234 \
        --data_seed 1234 \
        --gradient_checkpointing \
        --cache_dir ${cache_dir} \
        --output_dir ${output_dir} \
        --deepspeed configs/deepspeed_config_stage3_no_offload.json \
        --report_to ${report_to} \
        "${@}"
#        --use_flash_attention \
#         --resume_from_checkpoint ...
fi

#scancel ${SLURM_JOB_ID}