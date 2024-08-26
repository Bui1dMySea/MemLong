import os, sys
import torch
import argparse
import json
import faiss
from tqdm import tqdm
import random
from argparse import Namespace
import numpy as np
import itertools
from math import *
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from peft import PeftModel,AutoPeftModelForCausalLM,PeftConfig
from time import time
from datetime import datetime
from MemLong import modeling_llama_position  
from MemLong import configuration_llama
from MemLong.utils import ToolkitConfig
from longllama.configuration_longllama import LongLlamaConfig
from longllama.modeling_longllama import LongLlamaForCausalLM


def parse_config():
    parser = argparse.ArgumentParser(description="Arguments for evaluating in context learning(ICL)")
    parser.add_argument("--loc_data_dir",type=str,help="The path to the data directory")
    parser.add_argument("--model_provider",type=str,choices=['OpenLLaMA','LongLLaMA','MemLong',"RagOpenLLaMA"])
    parser.add_argument("--model_path_or_name",type=str,help="The path to the model checkpoint or the model name")
    parser.add_argument("--task",type=str,choices=['mpqa','mr','sst-2','sst-5','subj'],help="The task for in-context learning")
    parser.add_argument("--peft_model",type=str,default=None,help="The path to the PEFT model")
    parser.add_argument("--attn_implementation",type=str,choices=['eager','sdpa','flash_attention_2'],default='eager')
    parser.add_argument("--k",type=int,default=20,help="The number of demonstration examples")
    parser.add_argument("--cache_k",type=int,default=0,help="The number of cached examples in LongMem's memory")
    parser.add_argument("--subset", type=str, default="test", help="normally test set. But for SST-2, there is no testset, we use validation set instead")
    parser.add_argument("--last_context_length",default=0,type=int,help="The last context length for LongLLaMA")
    parser.add_argument("--memory_size",type=int,help="The memory size for MemLong")
    parser.add_argument("--embedder_name",type=str,choices=["llm_embedder",'st_embedder','bge_embedder'],help="The embedder name for MemLong")
    parser.add_argument("--output_dir",type=str,default="result",help="The path to the output directory")
    args = parser.parse_args()
    return args
    
def load_data(loc_data_dir,task):
    data = {}
    path = os.path.join(loc_data_dir, task)
    if task == "sst-2":
        for split in ["train", "validation",]:
            data[split] = []
            file_split = "dev" if split == "validation" else split
            with open(os.path.join(path, "{}.tsv".format(file_split)), "r") as f:
                for line in tqdm(f.readlines()):
                    review, label = line.strip("\n").split("\t")
                    if label == "label":
                        continue
                    label = "positive" if int(label) == 1 else "negative"
                    data[split].append([review, label])
    
    if task == "sst-5":
        mapping = {0: "terrible", 1: "bad", 2: "okay", 3: "good", 4: "great"}
        for split in ["train", "validation", "test"]:
            file_split = "dev" if split == "validation" else split
            if os.path.exists(os.path.join(path, "stsa.fine.{}".format(file_split))):
                data[split] = []
                with open(os.path.join(path, "stsa.fine.{}".format(file_split)), "r") as f:
                    for line in tqdm(f.readlines()):
                        review, label = line[2:], line[0]                        
                        label = mapping[int(label)]
                        data[split].append([review.strip("\n"), label])

    elif task in ["mr", "mpqa"]:
        for split in ["train", "validation", "test"]:
            if os.path.exists(os.path.join(path, "{}.csv".format(split))):
                data[split] = []
                with open(os.path.join(path, "{}.csv".format(split)), "r") as f:
                    for line in tqdm(f.readlines()):
                        review, label = line[2:], line[0]
                        label = "positive" if int(label) == 1 else "negative"
                        data[split].append([review.strip("\n"), label])
        print(data['test'][0])

    elif task in ["subj"]:
        for split in ["train", "test"]:
            if os.path.exists(os.path.join(path, "{}.csv".format(split))):
                data[split] = []
                with open(os.path.join(path, "{}.csv".format(split)), "r") as f:
                    for line in tqdm(f.readlines()):
                        review, label = line[2:], line[0]
                        label = "subjective" if int(label) == 0 else "objective"
                        data[split].append([review.strip("\n").strip('"'), label])
        print(data['test'][0])
    
    return data


def main():
    args = parse_config()
    task_template_dict = {"sst-2": "Review: {} Sentiment: {}. ",
                          "sst-5": "Review: {} Sentiment: {}. ",
                            "mr": "Review: {} Sentiment: {}. ",
                          "mpqa": "Review: {} Sentiment: {}. ",
                          "subj": "Input: {} Type: {}. "}
    # Config
    if args.model_provider in ["LongLLaMA"]:
        config = LongLlamaConfig.from_pretrained(args.model_path_or_name)
        config.last_context_length = args.last_context_length
    
    elif args.model_provider in ["MemLong"]:
        if args.peft_model:
            config = configuration_llama.LlamaConfig.from_pretrained(args.peft_model)
        else:
            config = configuration_llama.LlamaConfig.from_pretrained(args.model_path_or_name)
        config.use_cache = False
        if args.memory_size is not None:
            config.memory_size = int(args.memory_size)
        if args.last_context_length > 0:
            config.last_context_length = args.last_context_length
        else:
            args.last_context_length = config.last_context_length
        config.memory_size = 65536
            
    elif "peft" in args.model_provider:
        config = PeftConfig.from_pretrained(args.peft_model)
    else:
        config = AutoConfig.from_pretrained(args.model_path_or_name)
        
    if "OpenLLaMA" in args.model_provider:
        if args.peft_model:
            tokenizer = AutoTokenizer.from_pretrained(args.peft_model)
            model = AutoPeftModelForCausalLM.from_pretrained(
                    args.peft_model,
                    config=config,
                    use_flash_attention_2="flash_attention_2" if args.flash_attn else None,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
            ).eval()
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
            model = AutoModelForCausalLM.from_pretrained( 
                    args.model_path_or_name,  
                    device_map="cuda",  
                    attn_implementation=args.attn_implementation,
                    torch_dtype="auto",
                    trust_remote_code=True,  
                )             
        tokenzier_vocab_size = len(tokenizer)
        model_vocab_size = model.get_input_embeddings().weight.size(0)
        if model_vocab_size != tokenzier_vocab_size:
            assert tokenzier_vocab_size > model_vocab_size
            print("Resize model embeddings to fit tokenizer")
            model.resize_token_embeddings(tokenzier_vocab_size)
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            num_new_tokens = tokenzier_vocab_size - model_vocab_size
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        else:
            tokenizer.pad_token_id = config.pad_token_id
            
    elif args.model_provider == "LongLLaMA":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
        model = LongLlamaForCausalLM.from_pretrained(args.model_path_or_name,torch_dtype=torch.bfloat16,device_map="auto").eval()
        tokenizer.pad_token_id = config.pad_token_id
    elif args.model_provider == "MemLong":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
        if args.embedder_name == "llm_embedder":
            toolkit_config = ToolkitConfig(
                    task="lrlm",
                    embedder_name="llm_embedder",
                    embedder_path="/opt/data/private/lwj/acl2023/FineRange/llm-embedder",
                    ret_embeddings_dim=768,
                )
        elif args.embedder_name == "bge_embedder":
            toolkit_config = ToolkitConfig(
                task="lrlm",
                embedder_name="bge_embedder",
                embedder_path="/opt/data/private/lwj/emnlp2024/bge-m3",
                ret_embeddings_dim=1024,
            )
        elif args.embedder_name == "st_embedder":
            toolkit_config = ToolkitConfig(
                task="lrlm",
                embedder_name="st_embedder",
                embedder_path="/opt/data/private/lwj/emnlp2024/st-m3",
                ret_embeddings_dim=512,
            )
        else:
            raise NotImplemented
        model = modeling_llama_position.LlamaForCausalLM.from_pretrained(
            args.model_path_or_name,
            config=config,
            toolkit_config=toolkit_config,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch.bfloat16,
            # device_map="auto",
        )
        tokenzier_vocab_size = len(tokenizer)
        model_vocab_size = model.get_input_embeddings().weight.size(0)
        num_new_tokens = tokenzier_vocab_size - model_vocab_size
        if num_new_tokens > 0:
            print("Resize model embeddings to fit tokenizer")
            model = model.resize_token_embeddings(tokenzier_vocab_size)
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        else:
            tokenizer.pad_token_id = config.pad_token_id

        if args.peft_model:
            print("Loading PEFT model")
            model = PeftModel.from_pretrained(
                model,
                args.peft_model,
                #device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            model = model.merge_and_unload()
        model.set_toolkit_tokenizer(tokenizer)
        model.to("cuda").eval()
    
    data = load_data(args.loc_data_dir,args.task)
    if args.task in ['sst-2', "sst-5"]:
        args.subset = "validation"

    task_template = task_template_dict[args.task]
    context_length = 1024
    model_list_acc = []
    
    model = model.eval()
    model = model.cuda()
    seeds = [1,2,3]
    with tqdm(total=len(seeds)*len(data[args.subset])) as pbar:
        for seed in seeds:
            random.seed(seed)
            original_demon_train_subset = random.sample(data['train'], args.k)
            original_demon_train_subset = [task_template.format(s[0], s[1]) for s in original_demon_train_subset]
            demonstration = "".join(original_demon_train_subset)
            total_cnt = 0
            acc_cnt = 0
            print("Load {} examples into memory".format(args.cache_k))
            for item in data[args.subset]:
                total_cnt += 1
                mem_inputs = None
                if args.model_provider in ["LongLLaMA","MemLong"]:
                    memory_set = [task_template.format(s[0], s[1]) for idx, s in enumerate(random.sample(data['train'], args.cache_k))]
                    memory_set = "".join(memory_set)
                    mem_inputs = tokenizer(memory_set, return_tensors='pt', padding=True)
                    mem_input_ids = mem_inputs['input_ids'].to(model.device)
                    mem_attention_mask = mem_inputs['attention_mask'].to(model.device)
                elif args.model_provider == "RagOpenLLaMA" :
                    memory_set = [task_template.format(s[0], s[1]) for idx, s in enumerate(random.sample(data['train'], args.cache_k))]
                    memory_set = "".join(memory_set)

                test_subset = original_demon_train_subset + [task_template[:-5].format(item[0])]
                inputs = tokenizer(''.join(test_subset), return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(model.device)
                attention_mask = inputs['attention_mask'].to(model.device)

                if args.model_provider in ["LongLLaMA","MemLong"]:
                    with torch.no_grad():
                        if mem_inputs is not None:
                            prediction = model(torch.concat((mem_input_ids,input_ids),dim=-1), attention_mask=torch.concat((mem_attention_mask,attention_mask),dim=-1), use_cache=False,return_dict=True,last_context_length=input_ids.shape[-1]).logits          
                        else:
                            prediction = model(input_ids, attention_mask=attention_mask, use_cache=False,last_context_length=input_ids.shape[-1]+1,return_dict=True).logits  
                elif args.model_provider in ["RagOpenLLaMA"]:
                    with torch.no_grad():
                        if mem_inputs is not None:
                            prediction = model(torch.concat((mem_input_ids,input_ids),dim=-1), attention_mask=torch.concat((mem_attention_mask,attention_mask),dim=-1), use_cache=False,return_dict=True).logits          
                        else:
                            prediction = model(input_ids, attention_mask=attention_mask, use_cache=False,return_dict=True).logits  
                    
                elif args.model_provider == "OpenLLaMA":
                    if input_ids.shape[1] > 2048:
                        input_ids = input_ids[:,-2048:]
                        attention_mask = attention_mask[:,-2048:]
                    with torch.no_grad():
                        prediction = model(input_ids, attention_mask=attention_mask, use_cache=False,return_dict=True).logits
                prediction = prediction[0, -1, :].softmax(dim=-1)
                if  args.model_provider == "MemLong":
                    model.reset_memory()
                
                prediction = tokenizer.decode(prediction.argmax(-1).item())
                acc_cnt += (item[1].startswith(prediction.strip()) and prediction.strip() != "")
                pbar.update(1)
                
            model_list_acc.append(acc_cnt / total_cnt)

            try:
                if model.decoder.external_memory:
                    model.decoder.external_memory.reset()
            except AttributeError:
                pass

            print("Acc for random seed {}: {}".format(seed, acc_cnt / total_cnt))
    model_list_acc = [np.mean(model_list_acc), np.std(model_list_acc)] 
    
    print("Mean acc across 6 seeds: {:.4f}, std: {:.4f}".format(model_list_acc[0], model_list_acc[1]))
    
    location = f"{args.output_dir}/{args.model_provider}_icl.json"
    
    if not os.path.exists(args.output_dir):
        print("Creating output directory", args.output_dir)
        os.makedirs(args.output_dir)
    
    now = datetime.now()
    # 格式化日期和时间
    formatted_now = now.strftime("%Y-%m-%d %H:%M")
    info = {
        "k": args.k,
        "cache_k": args.cache_k,
        "task": args.task,
        "acc": model_list_acc[0],
        "std": model_list_acc[1],
        "peft model": args.peft_model,
        "time": formatted_now,
    }
        
    with open(f"{location}",'a+') as f:
        f.write(json.dumps(info) + "\n")

if __name__=="__main__":
    main()