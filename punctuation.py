
import csv
import datetime
import pandas as pd
import numpy as np
from glob import glob
from transformers import BertTokenizer
import torch
from torch import nn
import json
from tqdm import tqdm
from sklearn import metrics
import os
import sys
package_path = '/home/xinying/Speaker2/Chinese-Punctuation-Restoration-with-Bert-CNN-RNN'
sys.path.append(package_path)
import model_1_to_1
import datetime

def read_files(file_paths):
    lines = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines.extend(file.readlines())
    return lines

# def process_lines(lines):
#     for line in lines:
#         print(f"Processed: {line.strip()}")


input_files = sys.argv[1:]
lines = read_files(input_files)


from model_1_to_1 import (
    BertChineseLinearPunc,
    BertChineseCNNreplaceBert,
    BertChineseBigStrideCNN,
    BertChineseSlimCNNBert,
    BertChineseSlimCNNBertLSTM,
    BertChineseEmbSlimCNNBert,
    BertChineseEmbSlimCNNlstmBert,
    BertChineseEmbSlimCNNlstmBertLSTM,
)
glob('models/*')
path = '/home/xinying/Speaker2/Chinese-Punctuation-Restoration-with-Bert-CNN-RNN/models/20240624_143110'

with open(os.path.join(path, 'hyperparameters.json'), 'r') as f:
    hyperparameters = json.load(f)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

punctuation_enc = {
    'O': 0,
    '，': 1,
    '。': 2,
    '？': 3
}

segment_size = 200
device="cpu"
output_size = len(punctuation_enc)
dropout = hyperparameters['dropout']
# bert_punc = BertChineseEmbSlimCNNlstmBertLSTM(segment_size, output_size, dropout, None).to(device)
bert_punc = BertChineseLinearPunc(segment_size, output_size, dropout, None).to(device)
bert_punc.load_state_dict(torch.load(os.path.join(path, 'model')))
bert_punc.eval()

y_pred = []
y_true = []
inputs = lines[1]
tokens = tokenizer.tokenize(inputs)
x = tokenizer.convert_tokens_to_ids(tokens)
x=torch.tensor(x).unsqueeze(0)
output = bert_punc(x)
y_pred += list(output.argmax(dim=1).cpu().data.numpy().flatten())


# Modify tokens based on y_pred
for i in range(len(y_pred)):
    if y_pred[i] != 0:
        tokens[i] += list(punctuation_enc.keys())[y_pred[i]]
        
# Untokenize the output
# untokenized_output = tokenizer.convert_tokens_to_string(tokens)
# print(untokenized_output)
untokenized_output = ''.join(tokens)
# print(untokenized_output)


print("Speaker:", lines[0])
speaker = str(lines[0][:-1])
# import ipdb
# ipdb.set_trace()
with open(f'/home/xinying/Speaker2/Qwen-AIpendant/{speaker}.csv', 'a+', newline='') as csvfile:
    # 创建一个 CSV 写入对象
    csvwriter = csv.writer(csvfile)
    
    # 写入 CSV 文件的表头
    # csvwriter.writerow(['Timestamp', 'Message'])
    # timestamp = now.replace(minute=0, second=0, microsecond=0).isoformat()
    now = datetime.datetime.now()
    # 将 'now' 的分钟、秒和微秒设为 0
    iso_timestamp = now.replace(minute=0, second=0, microsecond=0).isoformat()

# 将T替换为空格，并将:00:00添加到末尾
    p = iso_timestamp.replace('T', ' ')[:-6] + ':00:00'
    
    # 需要写入的文字
    message = untokenized_output
    
    # 写入一行数据
    csvwriter.writerow([p, message])

print("CSV 文件创建并写入成功。")