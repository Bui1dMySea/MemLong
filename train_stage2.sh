# no use gate | low ret_attn_layers | larger mem granularity


# path
# if you skip the first stage, you should uncomment the following line
# model_name_or_path=openlm-research/open_llama_3b_v2
# if you want to continue the first stage, you should use the following line, otherwise, you should comment the following line
peft_model=MemLong-memory_size_32768-mem_group_size_128-seq_len_1024-topk_8-train_mode_lora-all_stage_1_lora_all
dataset=./data/processed/0.5B_1024

# embedding
embedder_path=BAAI/bge-m3
# llm_embedder | bge_embedder | st_embedder
embedder_name=bge_embedder
ret_embeddings_dim=1024

# MemLong Params
ret_attn_layers=(13 17 21 25)
last_context_length=1024
mem_layer=13
mem_group_size=128
ret_group_size=8

# Training Params
seq_len=1024
use_gate=False
memory_size=32768
train_mode=lora-freeze
main_process_port=26070
num_processes=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
batch_size=1
clear_memories_on_bos_token_id=False
num_train_epochs=1
date=$(date '+%m%d_%H%M')
targets=q_proj,k_proj,v_proj,o_proj
trainable_params=norm,embed
output_dir=stage_2_zero_parital_0.5B
if [ "$use_gate" = "True" ]; then
    trainable_params=ret_gate,norm,embed
else
    trainable_params=norm,embed
fi
output_dir=stage_2_zero_low_parital_larger_mem_granularity-0.5B

accelerate launch --config_file zero2_config.yaml --main_process_port ${main_process_port} --num_processes ${num_processes} run_clm_no_trainer.py \
--mem_group_size ${mem_group_size} \
--ret_attn_layers ${ret_attn_layers[*]} \
--output_dir ${output_dir} \
--per_device_train_batch_size ${batch_size} \
--per_device_eval_batch_size ${batch_size} \
--batch_sequential \
--num_train_epochs ${num_train_epochs} \
--load_from_disk ${dataset} \
--model_name_or_path ${model_name_or_path} \
--peft_model ${peft_model} \
--embedder_path ${embedder_path} \
--embedder_name ${embedder_name} \
--ret_embeddings_dim ${ret_embeddings_dim} \
--train_mode ${train_mode} \
--seq_len ${seq_len} \
--last_context_length ${last_context_length} \
--freeze_layers 0:$((mem_layer+1)) \
--num_warmup_steps 1000 \
--learning_rate 5e-5 \
--weight_decay 1e-4 \
--continual_finetuning \
--use_gpu_to_search \
--position_type Zero \
--project MemLong \
--project_name ${output_dir} \
--with_tracking \
--log_step 10 \
--trainable_params ${trainable_params} \
--targets ${targets}