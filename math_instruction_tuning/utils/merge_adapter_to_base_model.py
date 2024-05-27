from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
import argparse
import torch
import os
import json
from pathlib import Path
from safetensors import safe_open
from peft import (
    get_peft_model,
    PeftModel,
    VBLoRAConfig,
)
from peft.tuners.vblora import VBLoRALayer

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


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


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        # torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging the LoRA into the base model.")
    model = model.merge_and_unload()
    print("Saving the merged model to disk.")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Adapter to Base Model")
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--adapter", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    main(args)
