#!/bin/bash

# Configuration
category="items" # Changed from Toys_and_Games to a generic category
dataset="m_IOATBC-1.0-5-5"
data_path="/llm-reco-ssd-share/baohonghui/Baselines/RMDR/dataset"
BASE_MODEL="Qwen/Qwen2.5-1.5B" # Update this to your local model path

# Output setup
task_name="r3_rmdr_${category}"
out="./output_dir/${task_name}"
mkdir -p ${out}

# Distributed setup
port=$((RANDOM % 5000 + 10000))
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Update as needed

# Run training
# Using torchrun for distributed training (single node, 8 GPUs)
torchrun --nproc_per_node=8 --nnodes=1 --master_port=$port \
    src/latent/latent_attention_train.py \
    --base_model $BASE_MODEL \
    --use_rmdr True \
    --data_path $data_path \
    --dataset $dataset \
    --category $category \
    --output_dir ${out} \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --sample -1 \
    > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

# Generate items info file for evaluation
info_file="${out}/items.txt"
python src/utils/generate_r3_info.py --data_path $data_path --dataset $dataset --output_file ${info_file}

# Run evaluation
# Note: For simplicity, running on a single GPU. Can be parallelized similarly to latent_train.sh if needed.
python src/latent/latent_attention_eval.py \
    --base_model ${out} \
    --use_rmdr True \
    --data_path $data_path \
    --dataset $dataset \
    --category $category \
    --info_file ${info_file} \
    --result_json_data ${out}/eval_result.json \
    --sample -1 \
    --batch_size 32

# Calculate and print metrics (including domain-specific metrics)
python src/utils/calc.py \
    --path ${out}/eval_result.json \
    --item_path ${info_file}
