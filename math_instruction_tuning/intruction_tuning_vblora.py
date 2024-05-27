# Code based on https://github.com/GraphPKU/PiSSA script with minimal changes.

import copy
import os
import yaml
import json
import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal
from transformers.trainer_pt_utils import get_parameter_names
import torch
import transformers
from transformers import Trainer, set_seed
from datasets import load_dataset
from peft import (
    get_peft_model,
    VBLoRAConfig,
)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    adapter_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    dataset_split: str = field(
        default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"}
    )
    dataset_field: List[str] = field(
        default=None, metadata={"help": "Fields of dataset input and output."}
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    lora_r: int = field(
        default=None, metadata={"help": "The rank of incremental matrices."}
    )
    num_vectors: int = field(
        default=None,
        metadata={
            "help": "Number of vectors in the vector bank. Use higher values when the model size increases."
        },
    )
    vector_length: int = field(
        default=256,
        metadata={
            "help": "The length of the vectors in the vector bank. The length of the vectors should be divisible by the hidden dimension of the model."
        },
    )
    save_only_topk_weights: bool = field(
        default=False,
        metadata={
            "help": "Whether to only save the topk weights. Setting save_only_topk_weights = True significantly reduces storage space. However, models saved in this mode can be used for merging or inference only, not for resuming training."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [
        PROMPT.format_map(dict(instruction=instruction))
        for instruction in examples[query]
    ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    set_seed(script_args.seed)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map="auto",
    )

    if script_args.lora_r is not None:
        print(f"adding VB-LoRA modules...")
        modules = find_all_linear_names(model)
        config = VBLoRAConfig(
            r=script_args.lora_r,
            vector_length=script_args.vector_length,
            vblora_dropout=0,
            target_modules=modules,
            num_vectors=script_args.num_vectors,
            task_type="CAUSAL_LM",
            save_only_topk_weights=script_args.save_only_topk_weights,
        )

        model = get_peft_model(model, config)
    else:
        raise ValueError("LoRA rank should be provided.")

    now = datetime.datetime.now()
    now.strftime("%Y-%m-%dT%H:%M:%S") + ("-%02d" % (now.microsecond / 10000))

    adapter_name = "default"
    peft_config_dict = {adapter_name: config}

    script_args.output_dir = (
        f"{script_args.output_dir}/{script_args.model_name_or_path}/"
        f"{script_args.data_path}_split_{script_args.dataset_split}/"
        f"rank_{peft_config_dict[adapter_name].r}_lr_"
        f"{script_args.learning_rate}_seed_{script_args.seed}/output_{now}"
    )
    os.makedirs(script_args.output_dir)

    for param in model.parameters():
        param.data = param.data.contiguous()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_train_datasets = load_dataset(
        script_args.data_path, split=script_args.dataset_split
    )
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "query": script_args.dataset_field[0],
            "response": script_args.dataset_field[1],
        },
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)
    optimizer = create_optimizer(model, script_args)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=script_args,
        optimizers=(optimizer, None),
        **data_module,
    )
    model.config.use_cache = False

    print_trainable_parameters(model)

    trainer.train()
    trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir, "ft"))


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            if "logits" in name:
                num = param.numel() / param.shape[-1] * 3
            else:
                num = param.numel()
            print(name, param.dtype, param.shape, num)
            trainable_params += num
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def create_optimizer(model, args) -> torch.optim.Optimizer:
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    vector_bank_parameters = [
        name for name, _ in model.named_parameters() if "vector_bank" in name
    ]
    logits_parameters = [
        name for name, _ in model.named_parameters() if "logits" in name
    ]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n in decay_parameters
                and n not in logits_parameters
                and n not in vector_bank_parameters
            ],
            "weight_decay": 0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if n not in decay_parameters
                and n not in logits_parameters
                and n not in vector_bank_parameters
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n in vector_bank_parameters
            ],
            "lr": 0.001,
            "weight_decay": 0.0,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n in logits_parameters
            ],
            "lr": 0.01,
            "weight_decay": 0.0,
        },
    ]

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                skipped += sum(
                    {p.data_ptr(): p.numel() for p in module.parameters()}.values()
                )
                logger.info(f"skipped {module}: {skipped/2**20}M params")
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.debug(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"skipped: {skipped/2**20}M params")

    return optimizer


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


if __name__ == "__main__":
    train()
