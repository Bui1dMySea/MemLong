#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import faiss
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Sampler
from torch.cuda import device_count
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.utils import (
    DummyOptim,
    DummyScheduler,
)
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    SchedulerType,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from transformers.utils.versions import require_version
from transformers import AutoTokenizer
from huggingface_hub import Repository, create_repo
import datasets
from datasets import load_from_disk, load_dataset
from wandb.util import generate_id
from functools import partial
from src.utils import (
    BatchSequentialSampler,
    group_texts,
    save_config,
    ToolkitConfig,
    tokenize_fn,
    set_freeze_by_idxs,
    convert_to_lora,
)

from src.configuration_llama import LlamaConfig
from src.modeling_llama_position import LlamaForCausalLM
from peft import PeftModel

from deepspeed.runtime.zero.stage_1_and_2 import (estimate_zero2_model_states_mem_needs_all_live,)
from functools import partial

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

logger = get_logger(__name__)
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument("--load_from_disk",type=str,help="load dataset from disk")
    parser.add_argument("--embedder_path",type=str)
    parser.add_argument("--mem_layer",type=int,help="set the memory layer ")
    parser.add_argument("--ret_attn_layers",nargs="+",type=int,help="set the ret_attn_layers")
    parser.add_argument("--last_context_length",type=int,help="argument for inference")
    parser.add_argument("--batch_sequential",default=False,action="store_true")
    parser.add_argument("--clear_memories_on_eos_token_id", default=None, action="store_true")
    parser.add_argument("--clear_memories_on_bos_token_id", default=None, action="store_true")
    parser.add_argument("--circulate_steps",default=512,type=int)
    parser.add_argument("--ret_group_size", type=int, help="each group for retrieval")
    parser.add_argument("--pooling_tokens", default=None, required=False,type=int, help="compress the tokens size")
    parser.add_argument("--mem_group_size", type=int)
    parser.add_argument("--use_gate",action="store_true")
    parser.add_argument("--use_gpu_to_search", action="store_true")
    parser.add_argument("--trainable_params",default=None,type=str,)
    parser.add_argument("--targets",default=None, type=str)
    parser.add_argument("--use_cache", action="store_true",default=False)
    parser.add_argument("--position_type",choices=["Zero","Continual"],default=None)

    parser.add_argument("--task",type=str,default="lrlm",choices=["qa", "icl", "chat", "lrlm", "tool", "convsearch"],)
    parser.add_argument("--load_llama_config",type=str,default=None,)
    parser.add_argument("--dataset_name",type=str,default=None,help="The name of the dataset to use (via the datasets library).",)
    parser.add_argument("--dataset_config_name",type=str,default=None,help="The configuration name of the dataset to use (via the datasets library).",)
    parser.add_argument("--train_file",type=str,default=None,help="A csv, txt or a json file containing the training data.",)
    parser.add_argument("--validation_file",type=str,default=None,help="A csv, txt or a json file containing the validation data.",)
    parser.add_argument("--validation_split_percentage",default=5,help="The percentage of the train set used as validation set in case there's no validation split",)
    parser.add_argument("--model_name_or_path",type=str,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--config_name",type=str,default=None,help="Pretrained config name or path if not the same as model_name",)
    parser.add_argument("--tokenizer_name",type=str,default=None,help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--use_slow_tokenizer",action="store_true",help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",)
    parser.add_argument("--per_device_train_batch_size",type=int,default=8,help="Batch size (per device) for the training dataloader.",)
    parser.add_argument("--per_device_eval_batch_size",type=int,default=8,help="Batch size (per device) for the evaluation dataloader.",)
    parser.add_argument("--learning_rate",type=float,default=5e-5,help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",type=int,default=1,help="Total number of training epochs to perform.",)
    parser.add_argument("--max_train_steps",type=int,default=None,help="Total number of training steps to perform. If provided, overrides num_train_epochs.",)
    parser.add_argument("--embedder_name",type=str,default=None,)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=8,help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--lr_scheduler_type",type=SchedulerType,default="cosine_with_restarts",help="The scheduler type to use.",choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup",],)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--num_warmup_steps",type=int,default=1000,help="Number of steps for the warmup in the lr scheduler.",)
    parser.add_argument("--output_dir",type=str,required=True,help="Where to store the final model.",)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--model_type",type=str,default=None,help="Model type to use if training from scratch.",choices=MODEL_TYPES)
    parser.add_argument("--block_size",type=int,default=None,help=("Optional input sequence length after tokenization. The training dataset will be truncated in block of"" this size for training. Default to the model max input length for single sentence inputs (take into"" account special tokens)."),)
    parser.add_argument("--train_mode",type=str,default=None,required=True,choices=["lora-all", "lora-freeze", "partial-lora","partial-freeze"],)
    parser.add_argument("--continual_finetuning",action="store_true",)

    parser.add_argument("--freeze_layers", type=str,default=None)
    parser.add_argument("--update_boundary",type=str,default=None,)
    parser.add_argument("--memory_size",type=int,)
    parser.add_argument("--ret_embeddings_dim",type=int,default=768,)
    parser.add_argument("--preprocessing_num_workers",type=int,default=32,help="The number of processes to use for the preprocessing.",)
    parser.add_argument("--overwrite_cache",action="store_true",help="Overwrite the cached training and evaluation sets",)
    parser.add_argument("--no_keep_linebreaks",action="store_true",help="Do not keep line breaks when using TXT files.",)
    parser.add_argument("--push_to_hub",action="store_true",help="Whether or not to push the model to the Hub.",)
    parser.add_argument("--hub_model_id",type=str,help="The name of the repository to keep in sync with the local `output_dir`.",)
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--trust_remote_code",type=bool,default=False,help=("Whether or not to allow for custom models defined on the Hub in their own modeling files. This option""should only be set to `True` for repositories you trust and in which you have read the code, as it will ""execute code present on the Hub on your local machine."),)
    parser.add_argument("--checkpointing_steps",type=str,default=None,help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",)
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,help="If the training should continue from a checkpoint folder.",)
    parser.add_argument("--with_tracking",action="store_true",help="Whether to enable experiment trackers for logging.",)
    parser.add_argument("--report_to",type=str,default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_resume",type=str,default="auto",choices=["auto", "must", "never", "allow", None],)
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--low_cpu_mem_usage",action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    
    parser.add_argument("--load_best_model", action="store_true")
    parser.add_argument("--dataset_tokenizer", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--peft_model",type=str,default=None)
    args = parser.parse_args()
    if args.load_from_disk is not None:
        pass
    # Sanity checks
    elif (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError(
                "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
            )

    return args

def evaluate(args, model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(
            accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size))
        )

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss

def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    if args.with_tracking and args.log_step is None:
        args.log_step = 8

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            repo_id = create_repo(
                repo_name, exist_ok=True, token=args.hub_token
            ).repo_id
            # Clone repo locally
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            if args.debug:
                args.output_dir = args.output_dir + "_debug"
                os.makedirs(args.output_dir, exist_ok=True)
            else:
                os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.load_from_disk:
        lm_datasets = load_from_disk(args.load_from_disk)
        lm_datasets.set_format("torch")
    elif args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if not args.continual_finetuning:
        llama_config = LlamaConfig.from_pretrained(
            pretrained_model_name_or_path=args.model_name_or_path,
        )
        
        llama_config.mem_layer = args.mem_layer
        llama_config.ret_attn_layers = args.ret_attn_layers
        llama_config.memory_size = args.memory_size
        llama_config.last_context_length = args.last_context_length
        llama_config.ret_group_size = args.ret_group_size
        llama_config.use_gate=args.use_gate
        llama_config.update_boundary = args.update_boundary
        # llama_config.mem_positionals=args.mem_positionals
        llama_config.mem_group_size=args.mem_group_size
        llama_config.use_cache=args.use_cache
        save_config(llama_config, args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        toolkit_config = ToolkitConfig(
            # use_gpu_to_search=args.use_gpu_to_search,
            embedder_name=args.embedder_name,
            task=args.task,
            embedder_path=args.embedder_path,
            ret_embeddings_dim=args.ret_embeddings_dim,
        )
        model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path,config=llama_config, toolkit_config=toolkit_config,attn_implementation="eager")
        # model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path,config=llama_config,attn_implementation="eager")
    else:
        # peft priority
        if args.peft_model:
            llama_config = LlamaConfig.from_pretrained(args.peft_model+"/config.json")
        else:
            llama_config = LlamaConfig.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)
            
        llama_config.use_gate = llama_config.use_gate if getattr(args,"use_gate",None) == None else args.use_gate
        llama_config.use_cache =llama_config.use_cache if  getattr(args,"use_cache",None) == None else args.use_cache
        llama_config.ret_attn_layers = llama_config.ret_attn_layers if getattr(args,"ret_attn_layers",None) == None else args.ret_attn_layers
        llama_config.position_type = "Zero" if getattr(args,"position_type",None) == None else args.position_type
        llama_config.last_context_length= llama_config.last_context_length if getattr(args,"last_context_length",None) == None else args.last_context_length
        llama_config.mem_group_size = llama_config.mem_group_size if getattr(args,"mem_group_size",None) == None else args.mem_group_size
        llama_config.ret_group_size = llama_config.ret_group_size if getattr(args,"ret_group_size",None) == None else args.ret_group_size

        save_config(llama_config, args.output_dir)
        if not args.peft_model:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token_id = llama_config.pad_token_id
        toolkit_config = ToolkitConfig(
            # use_gpu_to_search=args.use_gpu_to_search,
            embedder_name=args.embedder_name,
            task=args.task,
            embedder_path=args.embedder_path,
            ret_embeddings_dim=args.ret_embeddings_dim,
        )
        model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path,config=llama_config, toolkit_config=toolkit_config,attn_implementation="eager")
        if args.peft_model:
            model = PeftModel.from_pretrained(
                model,
                args.peft_model,
                torch_dtype="bfloat16",
            )
            model = model.merge_and_unload()
            
    accelerator.print(llama_config)
    accelerator.print(toolkit_config)
    
    # total_params = sum(p.numel() for p in retree.parameters())
    # estimate_zero2_model_states_mem_needs_all_live(etree, num_gpus_per_node=8, num_nodes=1)
    model.set_toolkit_tokenizer(tokenizer)
    trainable_params = (args.trainable_params.split(",") if args.trainable_params else None)
    targets = args.targets.split(",") if args.targets else None
    lower, upper = list(map(int, args.freeze_layers.split(":")))
    # training mode
    if args.train_mode == "lora-all":
        model = convert_to_lora(model=model, targets=targets, trainable_params=trainable_params,task_type="CAUSAL_LM")
        model.print_trainable_parameters()
    elif args.train_mode == "lora-freeze":
        model = convert_to_lora(model=model, targets=targets, trainable_params=trainable_params)
        set_freeze_by_idxs(model, range(lower, upper),freeze=True)
        model.print_trainable_parameters()
    elif args.train_mode == "partial-lora":
        set_freeze_by_idxs(model,range(lower, upper),freeze=True)
        model = convert_to_lora(model=model, trainable_params=trainable_params)
        model.print_trainable_parameters()
    elif args.train_mode == "partial-freeze":
        set_freeze_by_idxs(model, range(lower, upper),freeze=True)
        if args.peft_model:
            model.print_trainable_parameters()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    # if len(tokenizer) > embedding_size:
    #    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    tokenized_datasets = None
    if not args.load_from_disk:
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        if args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > llama_config.max_position_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, llama_config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                block_size = min(1024, llama_config.max_position_embeddings)
        else:
            if args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(args.block_size, tokenizer.model_max_length)

        with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    if args.dataset_tokenizer:
        with accelerator.main_process_first():
            tokenized_datasets = lm_datasets.map(
                partial(tokenize_fn, tokenizer, args.seq_len),
                batched=True,
                batch_size=64,
                num_proc=56,
                remove_columns=lm_datasets["train"].column_names,
            )

    if tokenized_datasets is None:
        # breakpoint()
        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["validation"]
    else:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    # DataLoaders creation:
    train_dataloader = (
        DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            batch_size=args.per_device_train_batch_size,
            pin_memory=True,
        )
        if not args.batch_sequential
        else DataLoader(
            train_dataset["input_ids"],
            shuffle=False,
            collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            sampler=BatchSequentialSampler(
                train_dataset,
                args.per_device_train_batch_size,
                num_process=accelerator.num_processes,
            ),
            batch_size=args.per_device_train_batch_size,
            pin_memory=True,
        )
    )
    eval_dataloader = (
        DataLoader(
            eval_dataset,
            shuffle=True,
            collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            batch_size=args.per_device_eval_batch_size,
            pin_memory=True,
        )
        if not args.batch_sequential
        else DataLoader(
            eval_dataset["input_ids"],
            shuffle=False,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            ),
            sampler=BatchSequentialSampler(
                eval_dataset,
                args.per_device_eval_batch_size,
                num_process=accelerator.num_processes,
            ),
            batch_size=args.per_device_eval_batch_size,
            pin_memory=True,
        )
    )
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(
        # filter(lambda x: x.requires_grad is not False, retree.parameters()),
        optimizer_grouped_parameters,
        lr=args.learning_rate,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    accelerator.print("max_train_steps", args.max_train_steps)
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer,
            total_num_steps=args.max_train_steps,
            warmup_num_steps=args.num_warmup_steps,
        )
    # Prepare everything with our `accelerator`.
    (model,optimizer,train_dataloader,eval_dataloader,lr_scheduler,) = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        wandb_id = generate_id() if args.wandb_id is None else args.wandb_id
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        experiment_config["wandb_id"] = wandb_id
        accelerator.init_trackers(
            args.project,
            experiment_config,
            init_kwargs={
                "wandb": {
                    "name": args.project_name,
                    "resume": args.wandb_resume,
                    "id": wandb_id,
                }
            },
        )

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_metric_checkpoint = None

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # print trainable params
    if accelerator.is_local_main_process:
        logger.info(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
    for epoch in range(starting_epoch, args.num_train_epochs):
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        # logger.info("sample batch:", next(iter(active_dataloader)))
        if args.with_tracking:
            total_loss = 0
            losses = []
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs =  model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                    losses.append(
                        accelerator.gather_for_metrics(
                            loss.detach().float().repeat(args.per_device_eval_batch_size)
                        )
                    )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
            if args.with_tracking and completed_steps % args.log_step == 0:
                step_loss = torch.mean(torch.cat(losses))
                step_perplexity = math.exp(step_loss)
                accelerator.log(
                    {
                        "step_perplexity": step_perplexity,
                        "step_loss": step_loss,
                        # "epoch": epoch,
                        # "step": completed_steps,
                    },
                    step=completed_steps,
                )
                # losses = []

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break

        

        perplexity, eval_loss = evaluate(args, model, eval_dataloader, accelerator)
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "eval_perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss / len(train_dataloader),
                    "epoch": epoch,
                    # "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}",
                    blocking=False,
                    auto_lfs_prune=True,
                )

        if isinstance(checkpointing_steps, str) and checkpointing_steps == "epoch":
            accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}"))

        if best_metric is None or best_metric > perplexity:
            best_metric = perplexity
            best_metric_checkpoint = os.path.join(args.output_dir, "best_checkpoint")
            accelerator.save_state(best_metric_checkpoint)
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
            accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")

    
    if args.load_best_model:
        accelerator.load_state(best_metric_checkpoint)
        perplexity, eval_loss = evaluate(
            args, model, eval_dataloader, accelerator, eval_dataset
        )
        logger.info(
            f"Best model metrics: perplexity: {perplexity} eval_loss: {eval_loss}"
        )
        if perplexity != best_metric:
            raise AssertionError(
                f"Best metric {best_metric} does not match the metric {perplexity} of the loaded best model."
            )

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

        if accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity, "eval_loss": eval_loss.item()}, f)


if __name__ == "__main__":
    main()
