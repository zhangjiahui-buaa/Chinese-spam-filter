import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType
from pyspark.sql.functions import udf
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
import pandas as pd
import re
import jieba

max_iter = 100
with open('./stopwords.txt', encoding='utf8') as file:
    file_str = file.read()
    stopwords = file_str.split('\n')
    stopwords = set(stopwords)

def pre_proc(content):
    pattern = "[\u4e00-\u9fa5]+"
    regex = re.compile(pattern)
    mail_content = ''.join(regex.findall(content))
    cut_mail_content = (' '.join([k for k in jieba.lcut(mail_content, cut_all=True) if k not in stopwords]))
    return cut_mail_content

spark = SparkSession.builder.getOrCreate()
pandas_train_df = pd.read_csv('./train.csv')
pyspark_train_df = spark.createDataFrame(pandas_train_df)
pandas_test_df = pd.read_csv('./test.csv')
pyspark_test_df = spark.createDataFrame(pandas_test_df)
pyspark_train_df.show()
pyspark_test_df.show()

pre_proc_udf = udf(pre_proc,StringType())
pyspark_train_df = pyspark_train_df.withColumn("content",pre_proc_udf(pyspark_train_df["content"]))
pyspark_test_df = pyspark_test_df.withColumn("content",pre_proc_udf(pyspark_test_df["content"]))
pyspark_train_df.show()
pyspark_test_df.show()

tokenizer =Tokenizer(inputCol="content", outputCol="words")
hashingTF = HashingTF(inputCol = "words",outputCol = "rawfeatures")
idf = IDF(inputCol = "rawfeatures",outputCol = "features")
lr = LogisticRegression(maxIter = max_iter)

pipeline = Pipeline(stages = [tokenizer,hashingTF,idf,lr])

model = pipeline.fit(pyspark_train_df)
prediction = model.transform(pyspark_test_df).select("prediction").toPandas()
f = open('my_result.txt','w')
for i in range(len(prediction["prediction"])):
    f.write(str(int(prediction["prediction"][i]))+'\n')
f.close()








