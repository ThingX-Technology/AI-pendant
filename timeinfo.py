#conda activate eQW
import spacy
import dateparser
from datetime import datetime
import ipdb
import spacy
import numpy as np
import jieba
from fuzzywuzzy import fuzz
import csv
import pandas as pd
import spacy
from transformers import pipeline

nlp = spacy.load("zh_core_web_sm")
class TimeInfo:
    def __init__(self, query):
        self.query = query
        self.time = None
        self.date = "今天"
        self.AoP = "上午"
        self.hour = "9"
        # self.end_hour = "9"
        self.date_list = ["今天", "明天", "后天", "昨天", "前天"]
        self.AoP_list = ["上午", "下午", "晚上","早上"]
        self.AMhour_list = ["一点","两点","三点","四点","五点","六点","七点","八点","九点","十点","十一点","十二点"]
        # self.AMhour_list = ["1点","2点","3点","4点","5点","6点","7点","8点","9点","10点","11点","12点"]
        # self.PMhour_list = ["13点","14点","15点","16点","17点","18点","19点","20点","21点","22点","23点","24点"]
        self.parsed_date = None
        self.ch2ar = { 
            '一点': 1,
            '两点': 2,
            '三点': 3,
            '四点': 4,
            '五点': 5,
            '六点': 6,
            '七点': 7,
            '八点': 8,
            '九点': 9,
            '十点': 10,
            '十一点': 11,
            '十二点': 12,
        }
    def cal_sim(self, query, documents):
        query_vec = nlp(query).vector
        docs_vecs = np.array([nlp(doc).vector for doc in documents]) 
        #语义相似度得分
        cosine_similarities_vec = np.dot(docs_vecs, query_vec) / (np.linalg.norm(docs_vecs, axis=1) * np.linalg.norm(query_vec))
        fuzzy_score = [] #词汇相似度得分
        for doc, tokens in zip(documents, documents):
            # 计算每个文档与查询的模糊匹配得分
            score = max(fuzz.partial_ratio(" ".join(query), " ".join(tokens)) for token in tokens)*0.002
            fuzzy_score.append(score)
        fuzzy_score = np.array(fuzzy_score)
        # ipdb.set_trace()
        results = sorted(zip(cosine_similarities_vec + fuzzy_score, documents), reverse=True)
        best_res = results[0][1]
        return best_res
    def extract_time(self):
        doc = nlp(self.query)
        date_flag = False
        time_flag = False
        for ent in doc.ents:
            # ipdb.set_trace()
            if ent.label_ in ("DATE"):
                date_flag = True
                self.date = ent.text
                parsed_time = dateparser.parse(ent.text)
                self.parsed_date=str(parsed_time)[:11]
            elif ent.label_ in ("TIME"):
                time_flag = True
                self.AoP = self.cal_sim(ent.text, self.AoP_list)
                if self.AoP in ["上午", "早上"]:
                    self.hour = self.cal_sim(ent.text, self.AMhour_list)
                    self.hour = self.ch2ar[self.hour]
                    if len(str(self.hour)) == 1:
                        self.hour = "0"+str(self.hour)
                else:
                    self.hour = self.cal_sim(ent.text, self.AMhour_list)
                    self.hour = self.ch2ar[self.hour]+12
                    if len(str(self.hour)) == 1:
                        self.hour = "0"+str(self.hour)
            
        if not date_flag: #如果query中的token没有一个被识别为日期，则使用整句query匹配
            self.date = self.cal_sim(doc.text, self.date_list)
            parsed_time = dateparser.parse(self.date)
            self.parsed_date=str(parsed_time)[:11]
        if not time_flag: #如果query中的token没有一个被识别为time，则使用整句query匹配
            self.time = ent.text
            self.AoP = self.cal_sim(ent.text, self.AoP_list)
            if self.AoP in ["上午", "早上"]:
                    self.hour = self.cal_sim(ent.text, self.AMhour_list)
                    self.hour = self.ch2ar[self.hour]
                    if len(str(self.hour)) == 1:
                        self.hour = "0"+str(self.hour)
            else:
                self.hour = self.cal_sim(ent.text, self.AMhour_list)
                self.hour = self.ch2ar[self.hour]+12
                if len(str(self.hour)) == 1:
                    self.hour = "0"+str(self.hour)

    def parse_time(self):
        self.extract_time()
        if self.parsed_date != None:
            self.parsed_date+=f"{self.hour}:00:00"
        return self.parsed_date
    


def read_csv_and_search(file_path, search_key):
    """
    读取 CSV 文件并根据第一列的内容检索第二列的内容。

    参数:
    file_path (str): CSV 文件的路径
    search_key (str): 要检索的第一列的内容

    返回:
    str: 第二列的内容（如果找到），否则返回 None
    """
    try:
        # 读取 CSV 文件到 DataFrame
        df = pd.read_csv(file_path)
        # 检查搜索键是否在 DataFrame 的第一列中
        if search_key in df['timestamp'].values:
            # 获取对应的第二列的值
            result = df.loc[df['timestamp'] == search_key, 'text'].values[0]
            print("context:", result)
            return result
        else:
            return None  # 如果没有找到匹配的内容
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
# 示例查询
query = "今天上午十点找什么图标？"
#我今天晚上九点在干什么？
#我昨天早上十点去哪里了？
#我昨天早上十点和谁在一起？

ti=TimeInfo(query) # generate time stamp
time_stamp = ti.parse_time()
# ipdb.set_trace()
print("time stamp:", time_stamp)
file_path = '/home/xinying/Speaker2/Qwen-AIpendant/liyingn.csv'
search_key = time_stamp
context = read_csv_and_search(file_path, search_key)
nlp = spacy.load("zh_core_web_md")
qa_pipeline = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")
result = qa_pipeline({
    'question': query,
    'context': context
})

print(f"RoBERTa 模型 - 问题: {query}")
print(f"RoBERTa 模型 - 答案: {result['answer']}")

