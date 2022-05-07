import torch
from torch.utils.data import DataLoader,Dataset
from transformers import BertConfig,BertTokenizer,default_data_collator,DataCollatorForTokenClassification
import argparse
import random
import csv
from datasets_s.bert_dataset import BertDataset
from datasets_s.collate_functions import collate_to_max_length
from functools import partial
from model_own import GlyceBertDetectionForCsc
import os
from transformers import WEIGHTS_NAME

class CscDatasetPinyinZixingTest(Dataset):
    def __init__(self,args):
        super(CscDatasetPinyinZixingTest, self).__init__()
        self.args = args
        self.all_src = []
        self.all_tgt = []
        self.all_labels = []
        self.raw_datasets = self.get_rowdataset(args.test_file)
        self.tokenizer = BertDataset(args.model_name_or_path)
        self.detection_model = GlyceBertDetectionForCsc(args)
        state_dict = torch.load(os.path.join(args.detection_model_name_or_path, WEIGHTS_NAME))
        self.detection_model.load_state_dict(state_dict)


    def __getitem__(self, idx):
        input_ids, pinyin_ids = self.tokenizer.tokenize_sentence(self.raw_datasets[idx]['src'])
        input_shape = input_ids.size()
        attention_mask = torch.ones(input_shape)
        #pinyin_ids = pinyin_ids.view(input_shape[0], 8)

        logits = self.detection_model( input_ids=input_ids.view(1,input_shape[0]),
                                        pinyin_ids=pinyin_ids.view(1,input_shape[0],8),
                                        attention_mask=attention_mask.view(1,input_shape[0]))[0].squeeze()
        loss_mask = self.get_lossmask(logits=logits)
        input_ids_mask = torch.LongTensor([i if l==0 else 103 for i,l in zip(input_ids,loss_mask)])
        return input_ids,pinyin_ids,loss_mask,attention_mask,input_ids_mask



    def __len__(self):
        return len(self.raw_datasets)


    def get_rowdataset(self,filename):
        csvFile = open(filename, 'r')
        reader = csv.reader(csvFile )
        rawdataset = []
        for item in reader:
            temp = {}
            if reader.line_num == 1:
                continue
            temp['src'], temp['tgt'], temp['loss_mask'] = item[0].split('\t')
            rawdataset.append(temp)
            self.all_src.append(temp['src'])
            self.all_tgt.append(temp['tgt'])
            self.all_labels.append(temp['loss_mask'])
        csvFile.close()
        return rawdataset

    def get_lossmask(self,logits):
        predict_score = torch.softmax(logits, dim=-1)
        predict = []
        for score in predict_score:
            # predict.append(int(torch.argmax(word)))
            if score[1] >= self.args.error_value:
                predict.append(1)
            else:
                predict.append(0)
        return torch.tensor(predict)




def get_dataloder(args):
    dataset = CscDatasetPinyinZixingTest(args)
    dataloder =  DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=partial(collate_to_max_length,
                            fill_values=[0, 0, 0, 0, 0])
                            )
    return dataloder




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
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed." ),
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
        default=200,
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
        type=int,
        default=0.5,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.test_file = "data/csv_file/test_sighan15.csv"
    args.detection_model_name_or_path = "ChineseBERTMask_DetectionFromNNaddDence_weight1:5_epoch3"
    dataset = CscDatasetPinyinZixingTest(args)
    print(dataset[0])
    # print(len(dataset.all_src))
    # print(len(dataset.all_tgt))

    dataloder = get_dataloder(args)
    for data in dataloder:
        input_ids,pinyin_ids,loss_mask,attention_mask,input_ids_mask = data
        print(data)
        break

