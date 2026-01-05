#!/bin/bash

# Configuration
category="items" # Changed from Toys_and_Games to a generic category
dataset="m_IOATBC-1.0-5-5"
data_path="/Users/honghuibao/Desktop/Baselines/RMDR/dataset"
BASE_MODEL="/Users/honghuibao/.gemini/models/Qwen2.5-1.5B" # Update this to your local model path

# Output setup
task_name="r3_rmdr_${category}"
out="./output_dir/${task_name}"
mkdir -p ${out}

# Distributed setup
port=$((RANDOM % 5000 + 10000))
export CUDA_VISIBLE_DEVICES=0 # Update as needed

# Run training
# Using torchrun for distributed training (single node, one GPU for now as example)
torchrun --nproc_per_node=1 --nnodes=1 --master_port=$port \
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
    > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

# Optional: Evaluation part can be added here similar to latent_train.sh
