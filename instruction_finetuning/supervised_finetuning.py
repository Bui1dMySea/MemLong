import json
from dataclasses import asdict
from datetime import datetime
from peft import PeftModel
import torch
from src.modeling_llama_position import LlamaForCausalLM
from src.configuration_llama import LlamaConfig
from src.utils import ToolkitConfig,convert_to_lora,set_freeze_by_idxs
from transformers import HfArgumentParser, LlamaTokenizer, Trainer

from .arguments import DataArgs, ModelArgs, TokenizationArgs,CustomedTrainingArgs
from .data_processing import LOGGER, DataCollator, MixedTuneDataset
from .utils import get_packages, metrics_assign_group, non_numeric_to_str,smart_tokenizer_and_embedding_resize
import os

def main():
    hf_parser = HfArgumentParser((ModelArgs, DataArgs, TokenizationArgs, CustomedTrainingArgs))
    (
        model_args,
        data_args,
        tokenization_args,
        trainer_args,
    ) = hf_parser.parse_args_into_dataclasses()
    LOGGER.info(f"Preparing tokenizer {model_args.model_path}")
    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_path, padding_side="right", use_fast=False)
    
    if model_args.peft_path:
        LOGGER.info(f"Preparing Config {model_args.peft_path}")
        config = LlamaConfig.from_pretrained(model_args.peft_path + "/config.json")
    else:
        LOGGER.info(f"Preparing Config {model_args.model_path}")
        config = LlamaConfig.from_pretrained(model_args.model_path + "/config.json")
    config.use_cache = False
    toolkit_config = ToolkitConfig(
        task=model_args.embedder_task,
        embedder_name=model_args.embedder_name,
        embedder_path=model_args.embedder_path,
        ret_embeddings_dim=model_args.embedder_dim,
    )
        
    LOGGER.info(f"Preparing model {model_args.model_path}")
    model = LlamaForCausalLM.from_pretrained(model_args.model_path , config=config , toolkit_config = toolkit_config,attn_implementation=model_args.attn_implementation,torch_dtype=torch.bfloat16)
    
    tokenzier_vocab_size = len(tokenizer)
    model_vocab_size = model.get_input_embeddings().weight.size(0)
    num_new_tokens = tokenzier_vocab_size - model_vocab_size
    if model_vocab_size != tokenzier_vocab_size:
        model.resize_token_embeddings(tokenzier_vocab_size)
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    else:
        tokenizer.pad_token_id = config.pad_token_id
    model.set_toolkit_tokenizer(tokenizer)
    if model_args.peft_path:
        print("Loading PEFT model")
        model = PeftModel.from_pretrained(
            model,
            model_args.peft_path,
            # config=peft_config,
            # device_map="auto",
            torch_dtype="bfloat16",
        )
        model = model.merge_and_unload()
    
    trainable_params = (trainer_args.trainable_params.split(",") if trainer_args.trainable_params else None)
    targets = trainer_args.targets.split(",") if trainer_args.targets else None
    lower, upper = list(map(int, trainer_args.freeze_layers.split(":")))
    # training mode
    if trainer_args.train_mode == "lora-all":
        model = convert_to_lora(model=model, targets=targets, trainable_params=trainable_params,task_type="CAUSAL_LM")
        model.print_trainable_parameters()
    elif trainer_args.train_mode == "lora-freeze":
        model = convert_to_lora(model=model, targets=targets, trainable_params=trainable_params)
        set_freeze_by_idxs(model, range(lower, upper),freeze=True)
        model.print_trainable_parameters()
    elif trainer_args.train_mode == "partial-lora":
        set_freeze_by_idxs(model,range(lower, upper),freeze=True)
        model = convert_to_lora(model=model, trainable_params=trainable_params)
        model.print_trainable_parameters()
    elif trainer_args.train_mode == "partial-freeze":
        set_freeze_by_idxs(model, range(lower, upper),freeze=True)
        if model_args.peft_model:
            model.print_trainable_parameters()
        
    LOGGER.info("Preparing dataset")
    dataset = MixedTuneDataset(data_args=data_args, tokenizer=tokenizer, tokenization_args=tokenization_args)
    LOGGER.info("Preparing trainer")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=DataCollator(tokenizer=tokenizer,use_toolkit=model_args.use_toolkit),
    )

    model_args_dict = metrics_assign_group(asdict(model_args), "model_args")
    data_args_dict = metrics_assign_group(asdict(data_args), "data_args")
    tokenization_args_dict = metrics_assign_group(asdict(tokenization_args), "tokenization_args")
    trainer_args_dict = metrics_assign_group(asdict(trainer_args), "trainer_args")
    packages_dict = metrics_assign_group(get_packages(), "packages")
    all_params = {**model_args_dict, **data_args_dict, **tokenization_args_dict, **trainer_args_dict, **packages_dict}

    trainer.save_metrics("train", all_params, combined=True)

    str_params = json.dumps(all_params, indent=2)
    LOGGER.info(str_params)

    cur_time = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
    with open(f"{trainer_args.output_dir}/params_{cur_time}.json", "w") as f:
        f.write(str_params)

    LOGGER.info("Running trainer")

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=os.path.join(trainer_args.output_dir, "final_model"))


if __name__ == "__main__":
    main()