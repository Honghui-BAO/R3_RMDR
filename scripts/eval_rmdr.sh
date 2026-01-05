#!/bin/bash

# Configuration - Update these to match your training settings
category="items"
dataset="m_IOATBC-1.0-5-5"
data_path="/llm-reco-ssd-share/baohonghui/Baselines/RMDR/dataset"

# The path to the trained model checkpoint
# Default points to the output of the train_rmdr.sh script
task_name="r3_rmdr_${category}"
model_path="./output_dir/${task_name}"

# Output results
result_dir="${model_path}/evaluation"
mkdir -p ${result_dir}
info_file="${result_dir}/items.txt"
eval_json="${result_dir}/eval_result.json"

export CUDA_VISIBLE_DEVICES=0 # Typically eval runs on a single GPU

echo "=== Step 1: Generating items info file ==="
python3 -u src/utils/generate_r3_info.py \
    --data_path $data_path \
    --dataset $dataset \
    --output_file ${info_file}

echo "=== Step 2: Running model inference (Eval) ==="
# Using latent_attention_eval.py with RMDR support
python3 -u  src/latent/latent_attention_eval.py \
    --base_model ${model_path} \
    --use_rmdr True \
    --data_path $data_path \
    --dataset $dataset \
    --category $category \
    --info_file ${info_file} \
    --result_json_data ${eval_json} \
    --sample -1 \
    --batch_size 32

echo "=== Step 3: Calculating domain-specific metrics ==="
python3 -u src/utils/calc.py \
    --path ${eval_json} \
    --item_path ${info_file}
