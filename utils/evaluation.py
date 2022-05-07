import logging


class Metric():
    def __init__(self,src,tgt,pred,year):
        self.de = ['地','的','得']
        self.all_results = {}
        self.year = year
        truth_labels = self.process_sentence(src,tgt)
        pred_labels = self.process_sentence(src,pred)
        self.D_Results = self.detection_sentence_prf(truth_labels,pred_labels)
        self.C_Results = self.corection_sentence_prf(truth_labels,pred_labels)
        self.all_results.update(self.D_Results)
        self.all_results.update(self.C_Results)



    def process_sentence(self,src,tgt_or_pred):
        results = []
        for src_item,tgt_or_pred_item in zip(src,tgt_or_pred):
            # try:
            #     assert len(src_item) == len(tgt_or_pred_item)
            # except:
            #     print(src_item)
            #     print(tgt_or_pred_item)

            result = []
            for i,sen in enumerate(zip(src_item,tgt_or_pred_item)):
                if(sen[0]==sen[1]):
                    continue
                elif (sen[0]!=sen[1]) and sen[1] in self.de and self.year=="13":
                    continue
                else:
                    result.append((i+1,sen[1]))
            if(len(result) == 0):
                result.append((0,))
            results.append(result)
        return results


    #  opy 论文<Read, Listen, and See>
    # truth:[[(3, '气')], [0]]
    # pred: [[(1, '明')], [0]]
    def detection_sentence_prf(self,truth,pred):
        tp, targ_p, pred_p, hit = 0, 0, 0, 0
        for truth_item,pred_item in zip(truth,pred):
            if(len(truth_item[0]) > 1):
                targ_p += 1
            if(len(pred_item[0]) > 1):
                pred_p += 1
            if(len(truth_item)==len(pred_item) and all(p[0] == t[0] for t, p in zip(truth_item, pred_item))):
                hit += 1
            if(len(pred_item[0]) > 1 and len(truth_item)==len(pred_item) and  all(p[0] == t[0] for t, p in zip(truth_item, pred_item))):
                tp += 1

        acc = hit / len(truth)
        p = tp / pred_p
        r = tp / targ_p if targ_p > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        results = {
            'sent-detect-acc': round(acc * 100,2),
            'sent-detect-p': round(p * 100,2),
            'sent-detect-r': round(r * 100,2),
            'sent-detect-f1': round(f1 * 100,2),
        }
        return results

    def corection_sentence_prf(self,truth,pred):
        tp, targ_p, pred_p, hit = 0, 0, 0, 0
        for truth_item, pred_item in zip(truth, pred):

            if (len(truth_item[0]) > 1):
                targ_p += 1
            if (len(pred_item[0]) > 1):
                pred_p += 1
            if (len(truth_item) == len(pred_item) and all(p == t for t, p in zip(truth_item, pred_item))):
                hit += 1
            if (len(pred_item[0]) > 1 and len(truth_item) == len(pred_item) and all(
                    p == t for t, p in zip(truth_item, pred_item))):
                tp += 1

        acc = hit / len(truth)
        p = tp / pred_p
        r = tp / targ_p if targ_p > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        results = {
            'sent-corection-acc': round(acc * 100,2),
            'sent-corection-p': round(p * 100,2),
            'sent-corection-r': round(r * 100,2),
            'sent-corection-f1': round(f1 * 100,2),
        }
        return results


##########################################################
# 增加字符级别的探测和纠正指标  copy from wangdingming
# !/usr/bin/env python
# coding: utf-8
# @Author: Dimmy(wangdimmy@gmail.com)
# @Description: evaluation metrics script

def compute_prf(srcs,tgts,predicts):
    TP = 0
    FP = 0
    FN = 0
    all_predict_true_index = []
    all_gold_index = []
    for src, tgt, predict in zip(srcs, tgts, predicts):
        gold_index = []
        each_true_index = []
        for i in range(len(list(src))):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        all_gold_index.append(gold_index)
        predict_index = []
        for i in range(len(list(src))):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)



    # For the detection Precision, Recall and F1
    detection_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0
    logging.info(
        "The detection result is precision={}, recall={} and F1={}".format(detection_precision, detection_recall,
                                                                           detection_f1))
    TP = 0
    FP = 0
    FN = 0

    for i in range(len(all_predict_true_index)):
        # we only detect those correctly detected location, which is a different from the common metrics since
        # we wanna to see the precision improve by using the confusionset
        if len(all_predict_true_index[i]) > 0:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(predicts[i][j])
                if tgts[i][j] == predicts[i][j]:
                    TP += 1
                else:
                    FP += 1
            for j in all_gold_index[i]:
                if tgts[i][j] in predict_words:
                    continue
                else:
                    FN += 1

    # For the correction Precision, Recall and F1
    correction_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    correction_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall) if ( correction_precision + correction_recall) > 0 else 0
    logging.info("The correction  result is precision={}, recall={} and F1={}".format(correction_precision,correction_recall,correction_f1))

    return detection_precision, detection_recall,detection_f1, correction_precision,correction_recall,correction_f1


if __name__ == '__main__':
    src = ["民天天气怎么样？","民天天气怎么样？"]
    tgt = ["明天天气怎么样？","民天天气怎么样？"]
    pred = ["明天天气怎么样？","民天天气怎么样？"]
    print(compute_prf(src,tgt,pred))






















