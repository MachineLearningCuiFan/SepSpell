
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

from model_own import GlyceBertForCsc
from dataset import CscDatasetPinyinZixing,get_dataloder

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
    parser.add_argument(
        "--random_mask",
        type=int,
        default=5,
        help="表示每句话屏蔽句子长度的 1/random_mask 个字符.",
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

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

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
    if args.seed is not None:
        set_seed(args.seed)

    if args.model_name_or_path is not None:
        model = GlyceBertForCsc.from_pretrained(args.model_name_or_path)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)


    if args.train_file is not None:
        train_dataset = CscDatasetPinyinZixing(args)

    train_dataloader = get_dataloder(args)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    #######################          Train!        #############################
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  train batch size = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    tr_loss, logging_loss = 0.0, 0.0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_size, length = batch[0].shape
            pinyin_ids = batch[1].view(batch_size, length, 8)
            outputs = model(
                            input_ids=batch[0],
                            pinyin_ids=pinyin_ids,
                            label_ids=batch[2],
                            loss_mask=batch[3],
                            attention_mask=batch[4],
                            input_ids_mask=batch[5],
            )
            loss = outputs[0]
            tr_loss += loss.item()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if args.logging_steps > 0 and completed_steps % args.logging_steps == 0:
                logs = {}
                loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                learning_rate_scalar = scheduler.get_last_lr()[0]
                logs['learning_rate'] = learning_rate_scalar
                logs['loss'] = loss_scalar
                logging_loss = tr_loss
                print()
                print("Step: {}, LR: {}, Loss: {}".format(completed_steps, logs['learning_rate'], logs['loss']))

            if  args.save_steps > 0 and completed_steps % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, 'saved_ckpt-{}'.format(completed_steps))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                logger.info("Saving model checkpoint to %s", args.output_dir)
                # 如果我们有一个分布式模型，只保存封装的模型
                # 它包装在PyTorch DistributedDataParallel或DataParallel中
                model_to_save = model.module if hasattr(model, 'module') else model
                # 如果使用预定义的名称保存，则可以使用`from_pretrained`加载
                output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(output_dir, CONFIG_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(output_dir)
            if completed_steps >= args.max_train_steps:
                break

    if args.output_dir is not None:
        # 如果我们有一个分布式模型，只保存封装的模型
        # 它包装在PyTorch DistributedDataParallel或DataParallel中
        model_to_save = model.module if hasattr(model, 'module') else model
        # 如果使用预定义的名称保存，则可以使用`from_pretrained`加载
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)


if __name__ == "__main__":
    main()