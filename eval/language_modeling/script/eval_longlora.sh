#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "Working directory: $(pwd)"

batch_size=1   # context size during fine-tuning
num_context=0 # num previous chunks during evaluation
context_length=256 # per context length during evaluation
flash_attn=True
data_format=json
# 使用数组定义
dataset_path_or_name=(./data/json/wikitext103.json ./data/json/bookcorpus.json ./data/json/pg19.json ./data/json/proof_pile.json )
model_provider=LongLora
model_path_or_name=/opt/data/private/lwj/acl2023/llama-2-7b
peft_model=Yukang/Llama-2-7b-longlora-32k
num_proc=16
data_batch_size=16
seq_len=(1024 2048 4096 16384) # context length during evaluation

# 将数组转换为由空格分隔的字符串
dataset_paths="${dataset_path_or_name[*]}"

for len in "${seq_len[@]}"
do 
    echo "Evaluating Longlora model with sequence length $len"
    (
        python -u eval.py \
            --batch_size $batch_size \
            --context_size $len \
            --num_context $num_context \
            --context_length $context_length \
            --flash_attn $flash_attn \
            --data_format $data_format \
            --model_provider $model_provider \
            --peft_model $peft_model \
            --model_path_or_name $model_path_or_name \
            --dataset_path_or_name $dataset_paths \
            --num_proc $num_proc \
            --data_batch_size $data_batch_size \
            --seq_len $len \
            --sliding_window $len \
            --validation_load_strategy put_in_decoder 
    )
done
