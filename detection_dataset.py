import torch
from torch.utils.data import DataLoader,Dataset
from transformers import BertConfig,BertTokenizer,default_data_collator,DataCollatorForTokenClassification
import argparse
import random
import csv
from datasets_s.bert_dataset import BertDataset
from datasets_s.collate_functions import collate_to_max_length
from functools import partial

class CscDatasetPinyinZixing(Dataset):
    def __init__(self,args):
        super(CscDatasetPinyinZixing, self).__init__()
        self.args = args
        self.raw_datasets = self.get_rowdataset(args.train_file if args.train_file else args.test_file)
        self.tokenizer = BertDataset(args.model_name_or_path)


    def __getitem__(self, idx):
        input_ids, pinyin_ids = self.tokenizer.tokenize_sentence(self.raw_datasets[idx]['src'])
        tgt_ids,_ = self.tokenizer.tokenize_sentence(self.raw_datasets[idx]['tgt'])
        label_ids = torch.LongTensor(self.get_label_ids(input_ids,tgt_ids))
        input_shape = input_ids.size()
        attention_mask = torch.ones(input_shape)
        attention_mask[0] = 0
        attention_mask[-1] = 0
        return input_ids,pinyin_ids,label_ids,attention_mask

    def __len__(self):
        return len(self.raw_datasets)


    def get_label_ids(self,input_ids,tgt_ids):
        label_ids = []
        for s,t in zip(input_ids,tgt_ids):
            if s==t :
                label_ids.append(0)
            else:
                label_ids.append(1)
        return label_ids

    def get_rowdataset(self,filename):
        csvFile = open(filename, 'r')
        reader = csv.reader(csvFile )
        rawdataset = []
        for item in reader:
            temp = {}
            if reader.line_num == 1:
                continue
            temp['src'], temp['tgt'], temp['labels'] = item[0].split('\t')
            rawdataset.append(temp)
        csvFile.close()
        return rawdataset

def get_dataloder(args):
    dataset = CscDatasetPinyinZixing(args)
    dataloder =  DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True if args.train_file else False,
                            collate_fn=partial(collate_to_max_length,
                            fill_values=[0, 0, -100, 0,])
                            )
    return dataloder




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
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed." ),
    )

    parser.add_argument(
        "--model_name_or_path",
        default="ChineseBERT-base",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--random_mask",
        type=int,
        default=5,
        help="表示每句话屏蔽句子长度的 1/random_mask 个字符.",
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.test_file = "data/csv_file/test_sighan15.csv"
    dataset = CscDatasetPinyinZixing(args)
    #print(dataset.raw_datasets)


    dataloder = get_dataloder(args)
    for data in dataloder:
        input_ids,pinyin_ids,label_ids,attention_mask = data
        print(input_ids)
        print(label_ids)
        print(attention_mask)

        break

