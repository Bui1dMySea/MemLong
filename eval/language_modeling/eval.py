# Written by LIU WEIJIE
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
import json
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
# import InfiniteTransformer
# from LongLora import replace_llama_attn
from CEPE import LlamaForCausalContextLM
from datasets import Dataset
from time import time
from datetime import timedelta
from utils import ContextDataCollator,BatchSequentialSampler
from transformers.testing_utils import CaptureLogger
from peft import PeftModel,AutoPeftModelForCausalLM,PeftConfig
from MemLong import modeling_llama_position
from MemLong import configuration_llama
from MemLong.utils import ToolkitConfig,smart_tokenizer_and_embedding_resize
from transformers import DataCollatorForLanguageModeling
from longllama.configuration_longllama import LongLlamaConfig
from longllama.modeling_longllama import LongLlamaForCausalLM

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during inference')
    # parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=2048, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--context_length', type=int, default=0, help='number of transformer layers')
    parser.add_argument("--num_context",type=int,default=0,help="FOR CEPE ARUGMENT,SET 0 FOR CAUSAL MODEL")     # For CEPD
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    # parser.add_argument('--data_path', type=str, default="./test.bin", help='')
    parser.add_argument("--data_format", type=str, required=True, choices=["llama-bin", "json"])
    parser.add_argument("--dataset_path_or_name",type=str,nargs='+',required=True)
    parser.add_argument("--model_provider",type=str,choices=['LLaMA','Mistral','Gemma','CEPE',"InfiniteTransformer","LongLora","OpenLLaMA","MemLong","LongLLaMA","OpenLLaMA-peft","Phi-3-mini-128k-instruct","yarn-llama-2-7b-128k"])
    parser.add_argument("--model_path_or_name",type=str)
    parser.add_argument("--torch_dtype",type=str,default="bfloat16")
    parser.add_argument("--output_dir",type=str,default="./results")
    parser.add_argument("--num_proc",type=int,default=1)
    parser.add_argument("--data_batch_size",type=int,default=1000)
    parser.add_argument("--use_cache",default=False,action="store_true")
    parser.add_argument("--sliding_window",type=int,default=256)
    parser.add_argument("--sequential",action="store_true",help="DATA SEQUENCIAL EVALUATION")   
    parser.add_argument("--filter_length",type=int,default=32767,help="FOR CEPE ARUGMENT")     # For CEPD
    parser.add_argument("--segment_length",type=int,default=512,help="FOR INFINITRANSFORMER ARUGMENT")     # For InfiniteTransformer
    parser.add_argument("--last_context_length",type=int,default=-1,help="FOR MEMLONG MODEL")     # For MemLong
    parser.add_argument("--embedder_name",choices=['llm_embedder','st_embedder','bge_embedder'],default='llm_embedder',help="FOR LLaMA MODEL")     # For LLaMA
    parser.add_argument("--override",action="store_true",help="override result")     # For LLaMA
    parser.add_argument("--validation_load_strategy",choices=['put_in_decoder'],default=None) # None for CEPE, Normal is put_in_decoder
    parser.add_argument("--memory_size",default=None,type=int)
    parser.add_argument("--no_toolkit",action="store_true",default=False)
    parser.add_argument("--debug",action="store_true")
    parser.add_argument("--model_name",type=str,default=None)
    args = parser.parse_args()
    return args

tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

def tokenize_fn(tokenizer,text_column_name,examples,):
    with CaptureLogger(tok_logger) as cl:
        outputs = tokenizer(examples[text_column_name])
    # clm input could be much much longer than block_size
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
            " before being passed to the model."
        )
    return outputs

"""
def Sig_tokenize_fn(tokenizer, col_name, examples):
    outputs = tokenizer(
        (" "+ tokenizer.eos_token).join(examples[col_name]),
        truncation=False,
        padding=False,
        add_special_tokens=False,
    )
    return {"input_ids": outputs["input_ids"]}
"""
def preprocess(args,examples):
    # first calculate out the block size from n_ctx, ctx_size, and chunk_size and filter by length
    # second take chunks out with a stride of the eval sliding window
    # finally put things into the input_ids and encoder_input_ids if needed
    results = []
    for idx in range(len(examples["input_ids"])):
        input_ids = examples["input_ids"][idx]
        if len(input_ids) < args.filter_length:
            continue
        stride = args.sliding_window
        total_length = args.num_context * args.context_length + args.seq_len

        for i in range(0, len(input_ids) - total_length, stride):
            # we don't need the mask because we tokenized sequences without padding (the collator/forward func will handle the mask)
            ids = np.array(input_ids[i:i+total_length], dtype=np.int32)
            if "put_in_decoder" in args.validation_load_strategy:
                results.append({"input_ids": ids,})
            else:
                encoder_input_ids = ids[:args.num_context * args.context_length].reshape(args.num_context, args.context_length)
                ids = ids[args.num_context * args.context_length:]
                results.append({
                    "input_ids": ids, 
                    "encoder_input_ids": encoder_input_ids, 
                })
                
            labels = np.copy(ids).astype(np.int32)
            if stride < total_length:
                labels[:-stride] = -100
            results[-1]["labels"] = labels

            if args.filter_length > 32768:
                # only keep one sequence per document if the length is too long
                # otherwise we might store a lot of tokens with sliding window --> oom
                break

    results = {k: np.stack([d[k] for d in results]) for k in results[0]}
    return results

def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y

def iceildiv(x, y):
    return (x + y - 1) // y


def seqential_evaluate(model, eval_dataloader, args, device, tokenizer=None):
    stats = {}
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        all_losses = []
        shard_start = 0
        pbar = tqdm(eval_dataloader)
        for idx, batch in enumerate(pbar):
            if idx < shard_start:
                continue
        
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if args.model_provider == "MemLong":
                if args.no_toolkit:
                    batch['use_toolkit'] = False
                else:
                    batch["labels"][:,:-args.last_context_length] = -100
            
            if args.model_provider in ["LongLLaMA","MemLong"] and args.last_context_length:
                batch['last_context_length'] = args.last_context_length
                outputs = model(**batch)
            else:
                outputs = model(batch['input_ids'])
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            pbar.set_description(f"Loss: {loss.item()}")
            all_losses.append(loss.item())
        # mem_usage = sum([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
        eval_loss = torch.tensor(all_losses).mean().item()
        print(f"Eval loss: {eval_loss}")
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    print(f"val_perplexity: {perplexity}")
    stats['val_loss'] = eval_loss
    stats['val_perplexity'] = perplexity
    # stats['mem_usage'] = mem_usage
    return stats
    
def evaluate(model, data, batch_size, device, seq_length, sliding_window=256, use_cache=False,model_provider=None,segment_length=None,tokenizer=None):
    stats = {}
    model.eval()
    loss_list_val, acc_list = [], []
    loss_step_list_val = []
    with torch.no_grad():
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)
        for idx, (x, y) in tqdm(
            enumerate(get_as_batch(data['val'], seq_length, batch_size, device=device,sliding_window=sliding_window)),
            total=iceildiv(iceildiv(len(data['val']), sliding_window),batch_size)
        ):
            if model_provider in ['InfiniteTransformer']:
                input_ids = torch.tensor_split(x,list(range(segment_length,x.shape[1],segment_length)),dim=1)
                labels = torch.tensor_split(
                    x,
                    list(range(segment_length,x.shape[1],segment_length)),
                    dim=1
                )
                memory, norm_term = None,None
                for i in range(len(input_ids)):
                    with torch.no_grad():
                        outputs = model(
                            input_ids=input_ids[i],
                            labels=labels[i],
                            memory=memory,
                            norm_term=norm_term,
                        )
                    memory = outputs.memory
                    norm_term = outputs.norm_term
                loss = outputs.loss
                loss_list_val.append(loss.item())
            else:
                val_loss = 0.
                acc = 0.
                cnt = 0
                # breakpoint()
                # print(tokenizer.batch_decode(x,skip_special_tokens=True)[0])
                for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                    part_len = x[:, i:i + seq_length].shape[1]
                    outputs = model(input_ids=x[:, i:i + seq_length],labels=x[:, i:i+seq_length].contiguous(),use_cache=use_cache)
                    val_loss = outputs.loss * part_len + val_loss
                    acc = ((outputs.logits.argmax(-1) == y[:, i:i+seq_length]).float().sum()) + acc
                    cnt += part_len
                    while len(loss_step_list_val) <= part_idx:
                        loss_step_list_val.append([])
                    loss_step_list_val[part_idx].append(outputs.loss.item())
                val_loss /= cnt
                acc /= cnt
                loss_list_val.append(val_loss.item())
                acc_list.append(acc.item())

    #stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    
    #stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1))

    return stats

def main(args):
    seed = 2
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    #FIXME
    if args.data_format == "llama-bin":
        data = {'val': np.memmap(args.data_path, dtype=np.uint16, mode='r')}
        print(f"Num validation tokens: {len(data['val'])}")
        print("data path", args.data_path)
        print("base model", args.base_model)
        print("peft model", args.peft_model)
        if args.flash_attn and args.model_provider == "longlora":
            replace_llama_attn(use_flash_attn=True, use_full=True)
        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(args.base_model,cache_dir=args.cache_dir,)
        context_size = args.context_size if args.context_size > 0 else args.seq_len
        orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
        if orig_ctx_len and context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model,
            config=config,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.resize_token_embeddings(32001)

        if args.peft_model:
            trainable_params = os.path.join(args.peft_model, "trainable_params.bin")
            if os.path.isfile(trainable_params):
                model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
            else:
                raise ValueError("Trainable input embedding and normalization are required.")
            model = PeftModel.from_pretrained(
                model,
                args.peft_model,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        stats = evaluate(model, data, args.batch_size, device, args.seq_len, sliding_window=256)
        print(stats)

    elif args.data_format == "json":
        for dataset_path_or_name in args.dataset_path_or_name:
            dataset_name = dataset_path_or_name.split('/')[-1].split('.')[0]
            json_data = []
            with open(dataset_path_or_name) as file:
                for line in file:
                    json_data.append(json.loads(line))
            # load model and tokenizer
            if "/" in args.model_path_or_name:
                model_version = args.model_path_or_name.split("/")[-1]
            else:
                model_version = args.model_path_or_name
                
            if args.model_name is not None:
                location = f"{args.output_dir}/{args.model_name}/{args.seq_len}" if not args.debug else f"{args.output_dir}/{args.model_name}/{args.seq_len}_debug"
            else:
                location = f"{args.output_dir}/{args.model_provider}/{args.seq_len}" if not args.debug else f"{args.output_dir}/{args.model_provider}/{args.seq_len}_debug"

            
            if not os.path.exists(args.output_dir):
                print("Creating output directory", args.output_dir)
                os.makedirs(args.output_dir)
                
            if not os.path.exists(location):
                print("Creating directory", location)
                os.makedirs(location)

            if not args.override:
                if os.path.isfile(f"{location}/{dataset_name}_eval.json"):
                    print("File exists, skip evaluation")
                    continue

            # context scaling
            if args.model_provider in ['InfiniteTransformer']:
                config = InfiniteTransformer.GemmaConfig.from_pretrained(
                    args.model_path_or_name,
                    cache_dir=args.cache_dir,
                )
                config.segment_size = args.segment_length
            
            elif args.model_provider in ['MemLong']:
                if args.peft_model:
                    config = configuration_llama.LlamaConfig.from_pretrained(args.peft_model + "/config.json")
                    
                else:
                    config = configuration_llama.LlamaConfig.from_pretrained(args.model_path_or_name + "/config.json")
                config.use_cache = False
                if args.memory_size is not None:
                    config.memory_size = int(args.memory_size)
                if args.last_context_length > 0:
                    config.last_context_length = args.last_context_length
                else:
                    args.last_context_length = config.last_context_length

            elif args.model_provider in ['LongLLaMA']:
                config = LongLlamaConfig.from_pretrained(args.model_path_or_name)
                config.use_cache = args.use_cache
                config.last_context_length = args.last_context_length
            
            elif args.model_provider == "yarn-llama-2-7b-128k":
                from scaled_rope.configuration_llama import LlamaConfig
                from scaled_rope.modeling_llama_yarn import LlamaForCausalLM
                config = LlamaConfig.from_pretrained(args.model_path_or_name)
                config.rope_scaling = {
                    "type": "yarn",
                    "factor": 16,
                    "original_max_position_embeddings": 4096,
                }
            
            elif "peft" in args.model_provider :
                config = PeftConfig.from_pretrained(args.peft_model)
            else:
                config = transformers.AutoConfig.from_pretrained(args.model_path_or_name,trust_remote_code=True)
                
            context_size = args.context_size if args.context_size > 0 else args.seq_len
            orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
            if orig_ctx_len and context_size > orig_ctx_len and self.model_provider != "yarn-llama-2-7b-128k":
                scaling_factor = float(math.ceil(context_size / orig_ctx_len))
                config.rope_scaling = {"type": "linear", "factor": scaling_factor}
            
            if args.model_provider in ["LLaMA","OpenLLaMA","OpenLLaMA-peft","Phi-3-mini-128k-instruct"]:        
                if args.peft_model:
                    tokenizer = AutoTokenizer.from_pretrained(args.peft_model)
                    model = AutoPeftModelForCausalLM.from_pretrained(
                        args.peft_model,
                        config=config,
                        use_flash_attention_2="flash_attention_2" if args.flash_attn else None,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map="auto",
                    ).eval()
                else:
                    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_path_or_name,
                        config=config,
                        use_flash_attention_2="flash_attention_2" if args.flash_attn else None,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,  
                        device_map="auto",
                    ).eval()
                tokenzier_vocab_size = len(tokenizer)
                model_vocab_size = model.get_input_embeddings().weight.size(0)
                num_new_tokens = tokenzier_vocab_size - model_vocab_size
                if model_vocab_size < tokenzier_vocab_size:
                    print("Resize model embeddings to fit tokenizer")
                    model.resize_token_embeddings(tokenzier_vocab_size)
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
                collator = ContextDataCollator()
            elif args.model_provider == "yarn-llama-2-7b-128k":
                tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
                model = LlamaForCausalLM.from_pretrained(args.model_path_or_name,
                                    config=config,
                                    use_flash_attention_2="flash_attention_2" if args.flash_attn else None,
                                    torch_dtype=torch.bfloat16,
                                    trust_remote_code=True,  
                                    device_map="auto",
                                ).eval()
                collator = ContextDataCollator()
                
            elif args.model_provider == "Gemma":
                tokenizer = GemmaTokenizer.from_pretrained(args.model_path_or_name)
                model = GemmaForCausalLM.from_pretrained(
                    args.model_path_or_name,
                    config=config,
                    use_flash_attention_2="flash_attention_2" if args.flash_attn else None,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                ).eval()
                
                collator = ContextDataCollator()
                
            elif args.model_provider == "Mistral":
                tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
                model = MistralForCausalLM.from_pretrained(args.model_path_or_name,
                    config=config,
                    use_flash_attention_2="flash_attention_2" if args.flash_attn else None,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                ).eval()
                
                collator = ContextDataCollator()
            
            elif args.model_provider == "LongLLaMA":
                tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
                model = LongLlamaForCausalLM.from_pretrained(args.model_path_or_name,torch_dtype=torch.float32,device_map="auto").eval()
                collator = ContextDataCollator()
                
            elif args.model_provider == "LongLora":
                replace_llama_attn(use_flash_attn=args.flash_attn, use_full=True,inference=True)
                # Load model and tokenizer
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    args.model_path_or_name,
                    config=config,
                    cache_dir=args.cache_dir,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                model.resize_token_embeddings(32001)
                tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path_or_name)
                if args.peft_model:
                    trainable_params = os.path.join(args.peft_model, "trainable_params.bin")
                    if os.path.isfile(trainable_params):
                        model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
                    else:
                        raise ValueError("Trainable input embedding and normalization are required.")
                    model = PeftModel.from_pretrained(
                        model,
                        args.peft_model,
                        device_map="auto",
                        torch_dtype=torch.float16,
                    )
                
                collator = ContextDataCollator()
                
            elif args.model_provider == "InfiniteTransformer":
                # TODO
                tokenizer = GemmaTokenizer.from_pretrained(args.model_path_or_name)
                model = InfiniteTransformer.GemmaForCausalLM(
                    # args.model_path_or_name,
                    config=config,
                ).eval()
                if args.peft_model:
                    model = PeftModel.from_pretrained(
                        model,
                        args.peft_model,
                        torch_dtype=torch.bfloat16,
                    )
                model.to("cuda:0")    
                collator = ContextDataCollator()
                
            elif args.model_provider == "CEPE":
                # TODO
                tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
                model = LlamaForCausalContextLM.from_pretrained(
                    args.model_path_or_name,
                    config=config,
                    use_flash_attention_2="flash_attention_2" if args.flash_attn else None,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                ).eval()
                collator = ContextDataCollator()
                
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
                        embedder_path="/opt/data/private/lwj/memlong/bge-m3",
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
                    attn_implementation="flash_attention_2" if args.flash_attn else "eager",
                    cache_dir=args.cache_dir,
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
                model.to("cuda:0")
                model.eval()
                collator = ContextDataCollator()

            tokenizer.model_max_length = 1000000000000000019884624838656
            device = model.device    
            # tokenize
            if dataset_name == "wikitext103":
                dict_data = {'text': [''.join(item['text'] for item in json_data)]}
            else:
                dict_data = {"text": [item['text'] for item in json_data]}
            
            dataset = Dataset.from_dict(dict_data)
            tokenize_function = tokenize_fn
            tokenized_datasets = dataset.map(
                partial(tokenize_function, tokenizer, "text"), 
                batched=True,
                batch_size=args.data_batch_size,
                remove_columns=dataset.column_names,
                num_proc=args.num_proc
            )
            # FIXME
            # data = {"val":np.array(dataset["input_ids"],dtype=np.uint16)}
            
            # if tokenize_function == Sig_tokenize_fn:
            #     num_validation_tokens = len(np.array(dataset["input_ids"],dtype=np.uint16))
            #     data = {"val":np.array(dataset["input_ids"],dtype=np.uint16)}
            # else:
            tokenized_datasets = tokenized_datasets.filter(lambda example: len(example["input_ids"]) >= args.filter_length)
            eval_dataset = tokenized_datasets.map(
                partial(preprocess,args),
                batched=True,
                num_proc=args.num_proc,
                remove_columns=tokenized_datasets.column_names,
                batch_size=args.data_batch_size,
            )
            num_validation_tokens = sum([len(np.array(ele["input_ids"],dtype=np.uint16)) for ele in eval_dataset])
            # log
            print(f"Num validation tokens : {num_validation_tokens}")
            print("     data path  : ", dataset_path_or_name)
            print("     data name  : ", dataset_name)
            print("     base model : " , model_version)
            print("     peft model : ", args.peft_model)
            print(f"eval dataset size after filtering: {len(eval_dataset)}")   
            # print("Number of GPUs used : ", num_gpus_used) FIXME:get num_gpus_used
            info = {}
            info['Num validation tokens'] = num_validation_tokens
            info['data_path'] = dataset_path_or_name
            info['data_name'] = dataset_name
            info['base_model'] = model_version
            info['peft_model'] = args.peft_model
            # info['Number_of_GPUs_used'] = num_gpus_used
            
            eval_dataloader = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=args.batch_size,
                    collate_fn=collator,
                    pin_memory=True,
                )

            start_time = time()   
            """
            # CEPE Only support sequential evaluation     
            if args.model_provider in ["CEPE"]:
                eval_dataset = dataset.map(
                    partial(preprocess,args),
                    batched=True,
                    num_proc=args.num_proc,
                    remove_columns=dataset.column_names,
                    batch_size=args.data_batch_size,
                )
                
                eval_dataloader = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=args.batch_size,
                    collate_fn=collator,
                    pin_memory=True,
                    num_workers=args.num_proc,
                )
                stats = seqential_evaluate(model, eval_dataloader, args, device)
            elif args.model_provider in ["MemLong"]:
                # FIXME
                # eval_dataset = dataset.map(partial(preprocess,args),batched=True,num_proc=args.num_proc,remove_columns=dataset.column_names,batch_size=args.data_batch_size,)
                # eval_dataloader = torch.utils.data.DataLoader(eval_dataset,shuffle=False,collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False,),sampler=BatchSequentialSampler(eval_dataset,args.per_device_eval_batch_size),batch_size=args.batch_size,pin_memory=True,)
                # stats = seqential_evaluate(model, eval_dataloader, args, device)
                stats = evaluate(model, data, args.batch_size, device, args.seq_len, sliding_window=args.sliding_window,model_provider=args.model_provider,segment_length=args.segment_length,use_cache=False,tokenizer=tokenizer)
            else:
                stats = evaluate(model, data, args.batch_size, device, args.seq_len, sliding_window=args.sliding_window,model_provider=args.model_provider,segment_length=args.segment_length)
            """
            stats = seqential_evaluate(model, eval_dataloader, args, device,tokenizer=tokenizer)
            end_time = time()
            elapsed_seconds = end_time - start_time
            info['test_timestamp_duration'] = str(timedelta(seconds=elapsed_seconds))
    
            info = {**info,**stats}
            
            with open(f"{location}/{dataset_name}_eval.json", "w") as f:
                json.dump(info, f, indent=4)
                print("Evaluation results saved to", f"{location}/{dataset_name}_eval.json")
                
if __name__ == "__main__":
    args = parse_config()
    main(args)