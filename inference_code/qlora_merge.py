# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pdb

import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import (
    PeftModel
)

import os


def save_trained_model(base_model_path, peft_model_path, merged_model_path, trust_remote_code):

    print('Loading Base Model')

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code
    )

    tokenizer = AutoTokenizer.from_pretrained(
        peft_model_path,
        padding_side="right",
        truncation_side="left",
        use_fast=False,
        trust_remote_code=trust_remote_code
    )

    print('Loading Peft Model')

    peft_model = PeftModel.from_pretrained(model, model_id=peft_model_path, torch_dtype=torch.bfloat16)

    print('Merging Model')

    model = peft_model.base_model.merge_and_unload()

    print('Saving')

    model.save_pretrained(merged_model_path)

    tokenizer.save_pretrained(merged_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path to base model and peft model.')
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--peft_model', type=str, required=True)
    parser.add_argument('--save_merged_model_path', type=str, required=True)
    parser.add_argument('--trust_remote_code', type=bool, default=False, required=False)
    args = parser.parse_args()

    save_trained_model(args.base_model, args.peft_model, args.save_merged_model_path, args.trust_remote_code)

    config_path = os.path.join(args.save_merged_model_path, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        if config['max_position_embeddings'] < 4096:
            config['max_position_embeddings'] = 4096
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(config))
            print('set max_position_embeddings as 4096.')
        else:
            print('max_position_embeddings is already >= 4096')
