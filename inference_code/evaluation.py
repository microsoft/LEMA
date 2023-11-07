# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import argparse
import os
import pdb

from utils import get_final_result_gsm8k, get_final_result_math


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Path to gold file and inference file.')
    parser.add_argument('--test_data_folder', type=str, required=True)
    parser.add_argument('--test_data_file', type=str, required=True)
    parser.add_argument('--inference_folder', type=str, required=True)
    parser.add_argument('--inference_file', type=str, required=True)
    args = parser.parse_args()

    file_label = os.path.join(args.test_data_folder, args.test_data_file)

    file_pred = os.path.join(args.inference_folder, args.inference_file)

    print('file_label:', file_label)
    print('file_pred:', file_pred)

    acc_list = []
    with open(file_label, 'r', encoding='utf-8') as f_label, \
            open(file_pred, 'r', encoding='utf-8') as f_pred:
        labels = [json.loads(line)['completion'] for line in f_label.readlines()]
        preds = [json.loads(line)['samples'][0] for line in f_pred.readlines()]

        assert len(labels) == len(preds)

        for label, pred in zip(labels, preds):

            if 'gsm8k' in args.test_data_file:
                pred_result = get_final_result_gsm8k(pred)
                label_result = get_final_result_gsm8k(label)
            elif 'math' in args.test_data_file:
                pred_result = get_final_result_math(pred)
                label_result = get_final_result_math(label)
            else:
                pdb.set_trace()

            assert label_result is not None
            assert label_result != ''
            assert label_result != 0

            if pred_result == label_result:
                acc_list.append(1)
            else:
                acc_list.append(0)

    print('acc:', sum(acc_list) / len(acc_list))

