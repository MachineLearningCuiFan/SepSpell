
import argparse
import logging
import math
import os
import random
import torch
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    BertTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    WEIGHTS_NAME,
    CONFIG_NAME
)

from model_own import GlyceBertDetectionForCsc
from detection_dataset import CscDatasetPinyinZixing,get_dataloder

logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="ChineseBERT-base",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--error_value",
        type=float,
        default=0.5,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    # logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.


    if args.model_name_or_path is not None:
        model = GlyceBertDetectionForCsc(args)
        state_dict = torch.load(os.path.join(args.model_name_or_path, WEIGHTS_NAME))
        model.load_state_dict(state_dict)


    if args.test_file is not None:
        test_dataset = CscDatasetPinyinZixing(args)

    test_dataloader = get_dataloder(args)


    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    #######################          Test!        #############################
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  test batch size = {args.batch_size}")

    def evaluation(preds, refs):
        TP, FP, FN = 0, 0, 0
        all_predictions_index = []
        all_truth_index = []
        for idx, prediction_id in enumerate(preds):
            temp = []
            for i in range(len(prediction_id)):
                if prediction_id[i] == 1:
                    temp.append(i)
            all_predictions_index.append(temp)
        for truth_id in refs:
            temp = []
            for idx, num in enumerate(truth_id):
                if num == 1:
                    temp.append(idx)
            all_truth_index.append(temp)
        for idx, sentence in enumerate(all_predictions_index):
            for i in sentence:
                if i in all_truth_index[idx]:
                    TP += 1
                else:
                    FP += 1

        for idx, sentence in enumerate(all_truth_index):
            for i in sentence:
                if i in all_predictions_index[idx]:
                    continue
                else:
                    FN += 1
        P = TP / (TP + FP) if (TP + FP) > 0 else 0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * (P * R) / (P + R) if (P + R) > 0 else 0
        return P * 100, R * 100, F1 * 100

    def get_labels(predictions, references):
        true_predictions = [
            [p for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(predictions, references)
        ]
        true_labels = [
            [l for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(predictions, references)
        ]
        return true_predictions, true_labels

    all_refs = []
    all_pred = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            input_ids, pinyin_ids, label_ids, attention_mask = batch
            batch_size, length = input_ids.shape
            pinyin_ids = pinyin_ids.view(batch_size, length, 8)
            outputs = model(
                input_ids=input_ids,
                pinyin_ids=pinyin_ids,
                attention_mask=attention_mask,
                label_ids=None,
            )
            all_refs.extend(label_ids.tolist())
            logits = outputs[0]
            predict_score = torch.softmax(logits, dim=-1)

            for sentence in predict_score:
                predict = []
                for word in sentence:
                    # predict.append(int(torch.argmax(word)))
                    if word[1] >= args.error_value:
                        predict.append(1)
                    else:
                        predict.append(0)
                all_pred.append(predict)
    preds, refs = get_labels(all_pred, all_refs)

    print(evaluation(preds, refs))


if __name__ == "__main__":
    main()