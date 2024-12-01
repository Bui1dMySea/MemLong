import os
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain
from functools import partial
from tqdm import tqdm
import argparse
import math 

def parse_args():
    parser = argparse.ArgumentParser(description="Process data for MemLong")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for input_ids")
    parser.add_argument("--llama2_tokenizer", type=str, default="NousResearch/Llama-2-7b-chat-hf", help="Tokenizer for llama2")
    parser.add_argument("--hf_tokenizer", type=str,required=True, help="Tokenizer for huggingface")
    parser.add_argument("--slimpajama_name_or_path", type=str,required=True, help="Dataset name or path for slimpajama")
    parser.add_argument("--target_token", type=int, default=500000000,help="Target token for processing")

    return parser.parse_args()
    
def data_process(
    chunk_size: int,  
    llama2_tokenizer: AutoTokenizer,
    hf_tokenizer: AutoTokenizer,
    slimpajama_name_or_path: str,
    target_token: int = 500000000,
):
    if not os.path.exists(f"./processed/{target_token/1000000000:.1f}B_{chunk_size}"):
        os.makedirs(f"./processed/{target_token/1000000000:.1f}B_{chunk_size}")
        
    if hf_tokenizer.model_max_length < 131072:
        hf_tokenizer.model_max_length = 1310720    
    
    slimpajama_dataset = load_dataset(slimpajama_name_or_path)
    target_index = target_token // 131072 + 1
    partial_split = slimpajama_dataset["train"].select(range(target_index))
    column_names = list(partial_split.features)
    
    def _group_input_ids(examples, **kwargs):
        texts = llama2_tokenizer.batch_decode(examples["input_ids"])
        examples = hf_tokenizer(texts, add_special_tokens=False)
        concatenated_examples = {"input_ids": list(chain(*examples["input_ids"]))}
        total_length = len(concatenated_examples["input_ids"])
        total_length = (total_length // chunk_size) * chunk_size
        result = {
            k: [text[i:i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, text in concatenated_examples.items()
        }
        return result
    
    processed_chunks_set = partial_split.map(
        _group_input_ids,
        batch_size=4,
        batched=True,
        num_proc=64,
        remove_columns=column_names
    )
    processed_chunks_set = processed_chunks_set.train_test_split(train_size=0.95,seed=42,shuffle=False)
    processed_chunks_set["validation"] = processed_chunks_set['test']
    processed_chunks_set.pop("test")
    processed_chunks_set.save_to_disk(f"./processed/{target_token/1000000000:.1f}B_{chunk_size}")
    print("Downloaded and processed data")

if __name__ == "__main__":
    args = parse_args()
    data_process(
        chunk_size=args.chunk_size, 
        llama2_tokenizer=AutoTokenizer.from_pretrained(args.llama2_tokenizer),
        hf_tokenizer=AutoTokenizer.from_pretrained(args.hf_tokenizer),
        slimpajama_name_or_path=args.slimpajama_name_or_path,
        target_token=args.target_token,
    )

        
        
        