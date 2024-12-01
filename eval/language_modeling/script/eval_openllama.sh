#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "Working directory: $(pwd)"

batch_size=1
context_size=2048
flash_attn=True
data_format=json
# 使用数组定义
dataset_path_or_name=(./data/json/bookcorpus.json ./data/json/wikitext103.json ./data/json/proof_pile.json ./data/json/pg19.json )
model_provider=OpenLLaMA
model_path_or_name=/opt/data/private/lwj/open_llama_3b_v2
num_proc=16
data_batch_size=16
seq_len=(16384)

# 将数组转换为由空格分隔的字符串
dataset_paths="${dataset_path_or_name[*]}"

for len in "${seq_len[@]}"
do 
    echo "Evaluating open_llama_3b_v2 model with sequence length $len"
    (
        python -u eval.py \
            --batch_size $batch_size \
            --context_size $context_size \
            --flash_attn $flash_attn \
            --data_format $data_format \
            --model_provider $model_provider \
            --model_path_or_name $model_path_or_name \
            --dataset_path_or_name $dataset_paths \
            --num_proc $num_proc \
            --data_batch_size $data_batch_size \
            --seq_len $len \
            --override \
            --validation_load_strategy put_in_decoder \
    )
done
