
import argparse
import logging
import math
import os
import random
import torch
from tqdm.auto import tqdm
import pandas as pd

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

from model_own import GlyceBertForCsc
from test_dataset import CscDatasetPinyinZixingTest,get_dataloder
from utils.evaluation import Metric
from utils.evaluation import compute_prf
logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    # parser.add_argument(
    #     "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    # )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."),
    )

    parser.add_argument(
        "--model_name_or_path",
        default="ChineseBERT-base",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--detection_model_name_or_path",
        default="ChineseBERT-base",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--error_value",
        type=float,
        default=0.5,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )

    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")



    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
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

    if args.model_name_or_path is not None:
        model = GlyceBertForCsc.from_pretrained(args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)


    if args.test_file is not None:
        test_dataset = CscDatasetPinyinZixingTest(args)

    test_dataloader = get_dataloder(args)


    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    #######################          Test!        #############################
    logger.info("***** Running Testing *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Test batch size = {args.batch_size}")

    # Only show the progress bar once on each machine.
    model.eval()
    all_pred_word = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, pinyin_ids, loss_mask, attention_mask, input_ids_mask = batch
            batch_size, length = input_ids.shape
            pinyin_ids = pinyin_ids.view(batch_size, length, 8)
            outputs = model(
                            input_ids=input_ids,
                            pinyin_ids=pinyin_ids,
                            loss_mask=loss_mask,
                            attention_mask=attention_mask,
                            input_ids_mask=input_ids_mask,
            )
            logits = outputs[0]
            pred_id = torch.argmax(logits, dim=-1)
            batch_pred_word = []
            for pred,mask in zip(pred_id,loss_mask):
                preds = []
                pred_detec = []
                for t,l in zip(pred,mask):
                    if l == 1.0 :
                        pred_detec.append(tokenizer.convert_ids_to_tokens(int(t))+"F")
                        preds.append(tokenizer.convert_ids_to_tokens(int(t)))
                    else:
                        pred_detec.append("T")
                        preds.append('T')
                batch_pred_word.append(preds)
            all_pred_word.extend(batch_pred_word)


    all_src = test_dataset.all_src
    all_tgt = test_dataset.all_tgt

    all_pred = []
    for src,pred in zip(all_src,all_pred_word):
        src_id = tokenizer(src)['input_ids'][1:-1]
        pred = pred[1:len(src_id)+1]
        tokens = tokenizer.convert_ids_to_tokens(src_id)
        tokens = [t if not t.startswith('##') else t[2:] for t in tokens]
        tokens = [t if t != tokenizer.unk_token else 'U' for t in tokens]
        tokens_size = [len(t) for t in tokens]
        src_list = []
        index = 0
        for i in tokens_size:
            src_list.append(src[index:index + i])
            index += i
        for i,(s,p) in enumerate(zip(src_list,pred)):
            if p == 'T':
                continue
            elif p == '[UNK]':
                continue
            else:
                src_list[i] = p
        all_pred.append("".join(src_list))
    # 如果预测长度和实际长度不一致的情况  （这个问题是由于数据集中一些特殊字符导致）直接复制原句
    for i,(s,p) in enumerate(zip(all_src,all_pred)):
        if len(s) == len(p):
            continue
        else:
            all_pred[i] = s

    year = args.test_file.split("sighan")[-1][:2]
    metrics = Metric(all_src,all_tgt,all_pred,year)
    print(metrics.all_results)



if __name__ == "__main__":
    main()