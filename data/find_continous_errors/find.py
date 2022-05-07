import csv
import pandas as pd


def get_rowdataset(filename):
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

if __name__ == '__main__':
    all_datasets = []
    sighan13 = '../csv_file/test_sighan13.csv'
    sighan14 = '../csv_file/test_sighan14.csv'
    sighan15 = '../csv_file/test_sighan15.csv'
    for curpus_path in [sighan13,sighan14,sighan15]:
        all_datasets.extend(get_rowdataset(curpus_path))


    ##从三个测试集中找到所有连续错误，并写到csv文件中
    src = []
    tgt = []
    labels = []
    for data in all_datasets:
        for idx in range(len(data['labels'])-1):
            if data['labels'][idx] == data['labels'][idx+1] == "F":
                src.append(data['src'])
                tgt.append(data['tgt'])
                labels.append(data['labels'])
                break

    dataframe = pd.DataFrame({'src': src, 'tgt': tgt,'labels':labels})
    dataframe.to_csv("continuous_errors.csv",index=False,sep='\t')
    print(len(src))


