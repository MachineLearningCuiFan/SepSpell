
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
import os
import sys
# 获取当前代码文件绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将需要导入模块代码文件相对于当前文件目录的绝对路径加入到sys.path中
sys.path.append(os.path.join(current_dir, ".."))

from models.modeling_glycebert import GlyceBertForMaskedLM
from dataset import CscDatasetPinyinZixing,get_dataloder
from utils.evaluation import Metric,compute_prf


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
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

    parser.add_argument("--seed", type=int, default=233, help="A seed for reproducible training.")

    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )

    # add by cuifan  2021 8 20
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every X updates steps.",
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
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
        filename ="../realise_data_epoch5.log",
        datefmt="%m/%d/%Y %H:%M:%S",
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
    # if args.seed is not None:
    #     set_seed(args.seed)

    if args.model_name_or_path is not None:
        model = GlyceBertForMaskedLM.from_pretrained(args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)


    if args.test_file is not None:
        test_dataset = CscDatasetPinyinZixing(args)

    test_dataloader = get_dataloder(args)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    #######################          Test!        #############################
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  test batch size = {args.batch_size}")

    all_src = []
    all_tgt = []
    all_pred_id = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            batch_size, length = batch[0].shape
            pinyin_ids = batch[1].view(batch_size, length, 8)
            outputs = model(
                            input_ids=batch[0],
                            pinyin_ids=pinyin_ids,
            )
            logits = outputs[0]
            pred = torch.argmax(logits, dim=-1)
            all_pred_id.extend(pred)

        def get_all_pred(all_src, all_pred_id, tokenizer):
            all_pred = []
            for src, pred in zip(all_src, all_pred_id):
                pred_item = []
                src_bert = tokenizer(src, return_length=True)

                length = src_bert['length'] - 2

                pred_id = pred[1:length + 1].tolist()
                src_id = src_bert['input_ids'][1:length + 1]

                tokens = tokenizer.convert_ids_to_tokens(src_id)
                tokens = [t if not t.startswith('##') else t[2:] for t in tokens]
                tokens = [t if t != tokenizer.unk_token else 'U' for t in tokens]
                tokens_size = [len(t) for t in tokens]

                src_list = []
                index = 0
                for i in tokens_size:
                    src_list.append(src[index:index + i])
                    index += i

                for s, w, p in zip(src_id, src_list, pred_id):
                    if (671 <= s <= 7991):
                        pred_item.append(tokenizer.convert_ids_to_tokens(p))
                    else:
                        if (s == 100 and p != 100):
                            pred_item.append(tokenizer.convert_ids_to_tokens(p))
                        else:
                            pred_item.append(w)

                pred_item = [t if not t.startswith('##') else t[2:] for t in pred_item]
                pred_item = [t if t != tokenizer.unk_token else 'U' for t in pred_item]
                all_pred.append("".join(pred_item))
            return all_pred

        for data in test_dataset.raw_datasets:
            all_src.append(data['src'])
            all_tgt.append(data['tgt'])

        all_pred = get_all_pred(all_src, all_pred_id, tokenizer)


        with open("../pred.txt", 'w') as f:
            for sen in all_pred:
                f.write(sen + "\n")
        year = args.test_file.split("sighan")[-1][:2]
        print("############  句子级别的指标 ##############")
        metrics = Metric(all_src, all_tgt, all_pred, year)
        print(metrics.all_results)

        print("############  字符级别的指标 ##############")
        detection_precision, detection_recall, detection_f1, correction_precision, correction_recall, correction_f1 = compute_prf(
            all_src, all_tgt, all_pred)
        print(
            "detection_precision:{}, detection_recall:{},detection_f1:{}, correction_precision:{},correction_recall:{},correction_f1:{}".format(
                detection_precision, detection_recall, detection_f1, correction_precision, correction_recall,
                correction_f1))


if __name__ == "__main__":
    main()