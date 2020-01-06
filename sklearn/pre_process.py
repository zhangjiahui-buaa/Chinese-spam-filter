import pandas as pd
'''import findspark
findspark.init()
from pyspark.sql import SparkSession'''
import jieba
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import json
mail_list = []
cutword_list = []
label_list = []
test_cutwords_list=[]
with open('./stopwords.txt',encoding='utf8') as file:
    file_str = file.read()
    stopwords = file_str.split('\n')
    stopwords = set(stopwords)
def pre_proc(content,label):
    pattern="[\u4e00-\u9fa5]+"
    regex = re.compile(pattern)
    mail_content = ''.join(regex.findall(content))
    mail_list.append(mail_content)
    cutword_list.append(' '.join([ k for k in jieba.lcut(mail_content,cut_all = True) if k not in stopwords]))
    #print(cutword_list[-1])
    label_list.append(int(label))
def pre_proc_test(content):
    pattern="[\u4e00-\u9fa5]+"
    regex = re.compile(pattern)
    mail_content = ''.join(regex.findall(content))
    test_cutwords_list.append(' '.join([ k for k in jieba.lcut(mail_content,cut_all = True) if k not in stopwords]))

if __name__ == "__main__":
    
    #spark = SparkSession.builder.appName('main').getOrCreate()
    df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')
    start_time= time.time()
    df = df.apply(lambda row: pre_proc(row['content'],row['label']),axis =1)
    test_df = test_df.apply(lambda row:pre_proc_test(row['content']),axis = 1)
    print('time spent on pre_processing:',time.time()-start_time)


    #mail = spark.read.csv('./train.csv',header=True,inferSchema=True)
    cut_file = open('log.json','w',encoding='utf8')
    label_file = open('label.json','w',encoding='utf8')
    fp = open('test.json','w',encoding = 'utf8')
    json.dump(cutword_list,cut_file)
    json.dump(label_list,label_file)
    json.dump(test_cutwords_list,fp)
    fp.close()
    cut_file.close()
    label_file.close()

    
