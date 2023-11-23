# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import argparse
import os
import pdb

from utils import get_final_result_gsm8k, get_final_result_math, get_csqa_match


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
        label_infos = [json.loads(line) for line in f_label.readlines()]
        pred_infos = [json.loads(line) for line in f_pred.readlines()]

        assert len(label_infos) == len(pred_infos)

        for label_info, pred_info in zip(label_infos, pred_infos):

            if 'gsm8k' in args.test_data_file:
                pred = pred_info['samples'][0]
                label = label_info['completion']

                pred_result = get_final_result_gsm8k(pred)
                label_result = get_final_result_gsm8k(label)

                assert label_result is not None
                assert label_result != ''
                assert label_result != 0

                if pred_result == label_result:
                    acc_list.append(1)
                else:
                    acc_list.append(0)

            elif 'math' in args.test_data_file:
                pred = pred_info['samples'][0]
                label = label_info['completion']

                pred_result = get_final_result_math(pred)
                label_result = get_final_result_math(label)

                assert label_result is not None
                assert label_result != ''
                assert label_result != 0

                if pred_result == label_result:
                    acc_list.append(1)
                else:
                    acc_list.append(0)

            elif ('asdiv' in args.test_data_file) or ('mawps' in args.test_data_file) or ('svamp' in args.test_data_file):
                pred = pred_info['samples'][0]
                label = label_info['completion']

                pred_result = get_final_result_gsm8k(pred)
                label_result = get_final_result_gsm8k(label)

                pred_result = round(pred_result, 2)
                label_result = round(label_result, 2)

                if pred_result == label_result:
                    acc_list.append(1)
                else:
                    acc_list.append(0)

            elif 'csqa' in args.test_data_file:
                pred = pred_info['samples'][0]
                label = label_info['answer']
                candidates = label_info['candidates']

                assert label in candidates

                score = get_csqa_match(pred, label, candidates)
                acc_list.append(score)

            else:
                pdb.set_trace()

    print('acc:', sum(acc_list) / len(acc_list))

