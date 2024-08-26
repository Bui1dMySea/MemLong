from dataclasses import dataclass
import faiss
import torch
from torch.utils.data import Sampler
from typing import Dict, Iterator, Literal, Iterable, Union, List , Optional
from transformers import AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig
from itertools import chain
import os
import transformers
from .configuration_llama import LlamaConfig

TASK_CHOICE = Literal["qa", "icl", "chat", "lrlm", "tool", "convsearch"]


class BatchSequentialSampler(Sampler):
    def __init__(self, data, batch_size, num_process=None) -> None:
        self.total_len = len(data)
        # self.data = data[: self.total_len - self.total_len % batch_size]
        self.batch_size = batch_size
        self.num_process = num_process

    def __iter__(self) -> Iterator[int]:
        if self.num_process is None:
            per_batch_nums = self.total_len // self.batch_size
            for i in range(per_batch_nums):
                for j in range(self.batch_size):
                    yield i + j * per_batch_nums
        else:
            per_batch_nums = self.total_len // self.batch_size // self.num_process
            for i in range(per_batch_nums):
                for j in range(self.batch_size):
                    for k in range(self.num_process):
                        yield i + j * per_batch_nums + k * per_batch_nums * self.batch_size

    def __len__(self,) -> int:
        return self.total_len
    
def set_freeze_by_idxs(model, idxs: Union[int, List[int]], freeze: bool):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]

    if "llama" in type(model).__name__.lower():
        iter_model = list(list(model.children())[0].children())[1]
        num_child = len(iter_model)
    elif "peft" in type(model).__name__.lower():
        iter_model = list(list(list(list(model.children())[0].children())[0].children())[0].children())[-1]
        num_child = len(iter_model)
    else:
        iter_model = list(model.children())
        num_child = len(iter_model)
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(iter_model):
        if idx not in idxs:
            continue
        for name, param in child.named_parameters():
            if "peft" in type(model).__name__.lower():
                if "lora" in name or "modules_to_save" in name:
                    param.requires_grad = not freeze
                    
def group_texts(block_size, examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def tokenize_fn(tokenizer, seq_len, examples):
    tokenizer.padding_side = "right"
    text = list(map(lambda x: x + " " + tokenizer.eos_token,examples["raw_content"],))
    outputs = tokenizer(
        text,
        truncation=True,
        max_length=seq_len,
        return_overflowing_tokens=True,
        padding="max_length",
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == seq_len:
            input_batch.append(input_ids)

    return {"input_ids": input_batch}

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
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
        
def convert_to_lora(
    model,
    trainable_params: List[str] = None,
    targets: List[str] = None,
    r=64,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
):
    # Check the model type
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=targets,
        lora_dropout=lora_dropout,
        bias=bias,
        modules_to_save=trainable_params,
        task_type=task_type,
    )
    model = get_peft_model(model, lora_config)
    return model

def save_config(config, output_dir):
    config.save_pretrained(output_dir)

def load_config(config_path):
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")
    return LlamaConfig.from_pretrained(config_path)

@dataclass
class ToolkitConfig:
    task: TASK_CHOICE
    embedder_name: str
    embedder_path: str
    ret_embeddings_dim: int
    dtype: torch.dtype = torch.bfloat16
    
@dataclass
class MemConfig:
    positionals: bool
    use_gate: bool
    cache_dtype: torch.dtype
    
@dataclass
class MemCache:
    texts: List[Optional[List[str]]] = None
    embeddings:List[Union[faiss.IndexFlatIP,List[torch.Tensor]]] = None
    keys: torch.Tensor = None
    values: torch.Tensor = None
    masks: torch.Tensor = None

@dataclass
class SigMemCache:
    texts: List[str]
    embeddings: List[torch.Tensor]
    keys: torch.Tensor
    values: torch.Tensor 
    masks: torch.Tensor
    length: int
