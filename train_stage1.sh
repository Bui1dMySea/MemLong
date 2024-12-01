#!/bin/bash

# path
model_name_or_path=openlm-research/open_llama_3b_v2
dataset=./data/processed/0.5B_1024

# embedding
embedder_path=BAAI/bge-m3
# llm_embedder | bge_embedder | st_embedder
embedder_name=bge_embedder
ret_embeddings_dim=1024

# MemLong Params
ret_attn_layers=(14 15 16 17 18 19 20 21 22 23 24 25)
last_context_length=1024
mem_layer=13
mem_group_size=128
ret_group_size=8

# Training Params
seq_len=1024
use_gate=True
memory_size=32768
train_mode=lora-all
main_process_port=26070
num_processes=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
batch_size=1
clear_memories_on_bos_token_id=True
num_train_epochs=1
date=$(date '+%m%d_%H%M')
targets=q_proj,k_proj,v_proj,o_proj
trainable_params=norm,embed
output_dir=MemLong-memory_size_${memory_size}-mem_group_size_${mem_group_size}-seq_len_${seq_len}-topk_${ret_group_size}-train_mode_${train_mode}_stage_1_lora_all

accelerate launch --config_file zero2_config.yaml --main_process_port ${main_process_port} --num_processes ${num_processes} run_clm_no_trainer.py \
--model_name_or_path ${model_name_or_path} \
--load_from_disk ${dataset} \
--output_dir ${output_dir} \
--embedder_path ${embedder_path} \
--embedder_name ${embedder_name} \
--ret_embeddings_dim ${ret_embeddings_dim} \
--ret_attn_layers ${ret_attn_layers[*]} \
--last_context_length ${last_context_length} \
--mem_layer ${mem_layer} \
--mem_group_size ${mem_group_size} \
--ret_group_size ${ret_group_size} \
--seq_len ${seq_len} \
--memory_size ${memory_size} \
--train_mode ${train_mode} \
--per_device_train_batch_size ${batch_size} \
--per_device_eval_batch_size ${batch_size} \
--batch_sequential \
--freeze_layers 0:$((mem_layer + 1)) \
--num_train_epochs ${num_train_epochs} \
--clear_memories_on_bos_token_id \
--use_cache \
--update_boundary 10,90 \
--targets ${targets} \
--trainable_params ${trainable_params} \
--project MemLong \
--project_name ${output_dir} \
--with_tracking \
--log_step 10