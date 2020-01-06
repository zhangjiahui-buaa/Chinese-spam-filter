import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import json
if __name__ == "__main__":
    cutword = []
    label = []
    fp = open('log.json','r',encoding='utf8')
    cutword = json.load(fp)
    fp = open('label.json','r',encoding='utf8')
    label = json.load(fp)
    testset = json.load(open('test.json','r',encoding='utf8'))

    tfidf = TfidfVectorizer().fit(cutword+testset)
    X=tfidf.transform(cutword)
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(label)


    train_X,test_X,train_y,test_y = train_test_split(X,y,test_size= 0.0001)
    #train_X,test_X = preprocessing.scale(train_X,with_mean=False),preprocessing.scale(test_X,with_mean=False)
    model = LogisticRegressionCV(max_iter=300)
    model.fit(train_X,train_y)

    testset = tfidf.transform(testset)
    a = model.predict(testset)
    ffp = open('new_result.txt','w')
    for i in range(len(a)):
        ffp.write(str(a[i]))
        ffp.write('\n')
    ffp.close()
    model.score(test_X,test_y).round(4)