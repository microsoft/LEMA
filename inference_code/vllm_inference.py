# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional
import logging, os, json
from vllm import LLM, SamplingParams
import ray
from ray_on_aml.core import Ray_On_AML
import argparse


def inference(testdata_folder, testdata_file, output_folder, output_file, merged_model_path, tensor_parallel_size, trust_remote_code):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # log args
    logger.info(f"test file: {testdata_file}")
    logger.info(f"output file: {output_file}")
    logger.info(f"tensor_parallel_size: {tensor_parallel_size}")

    with open(os.path.join(testdata_folder, testdata_file), 'r', encoding='utf-8') as f_read:
        test_prompts = [json.loads(line)['prompt'] for line in f_read.readlines()]
        total_lines = len(test_prompts)
        logger.info(f"Total lines: {total_lines}")
    assert len(test_prompts) != 0

    llm = LLM(model=merged_model_path,
              tensor_parallel_size=tensor_parallel_size,
              trust_remote_code=trust_remote_code,
              max_num_batched_tokens=8192,
              gpu_memory_utilization=0.8)

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2048)

    batch_size = 128
    total_batch_num = (total_lines // batch_size) + 1

    current_lines = 0
    all_outputs = []

    for batch_idx in range(total_batch_num):
        if batch_idx == total_batch_num-1:
            prompt_batch = test_prompts[batch_idx * batch_size:]
        else:
            prompt_batch = test_prompts[batch_idx*batch_size:(batch_idx+1)*batch_size]
        results = llm.generate(prompt_batch, sampling_params)
        current_lines += batch_size
        logger.info(f"{current_lines} in {total_lines} examples.")
        for result in results:
            all_outputs.append({'samples': [result.outputs[0].text]})

    with open(os.path.join(output_folder, output_file), "w", encoding='utf-8') as f:
        for output in all_outputs:
            f.write(json.dumps(output) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for running vllm')
    parser.add_argument('--testdata_folder', type=str, required=True)
    parser.add_argument('--testdata_file', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--output_file', type=str, default="sample.jsonl", required=False)
    parser.add_argument('--merged_model_path', type=str, required=True)
    parser.add_argument('--tensor_parallel_size', type=int, default=4, required=False)
    parser.add_argument('--trust_remote_code', type=bool, default=False, required=False)
    args = parser.parse_args()

    inference(testdata_folder=args.testdata_folder,
              testdata_file=args.testdata_file,
              output_folder=args.output_folder,
              output_file=args.output_file,
              merged_model_path=args.merged_model_path,
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=args.trust_remote_code)
