import warnings
warnings.filterwarnings("ignore")
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


train=pd.read_csv('my_dataset.csv')
train.loc[train["Sentiment"]=='love','Sentiment']=1
train.loc[train["Sentiment"]=='sad','Sentiment']=0
# print(train.head())
x=train['Document']
y=train['Sentiment'].astype(int)

stop_words=set(stopwords.words('english'))
for i in range(len(x)):
	sentence=x.iloc[i]
	sentence=word_tokenize(sentence)
	filterd_sentence=[]
	# for words in sentence if not words in stop_words:
	# 	filterd_sentence.append(words)
	filterd_sentence=[w for w in sentence if not w in stop_words]
	filterd_sentence=' '.join(filterd_sentence)
	x.iloc[i]=filterd_sentence

vectorizer=TfidfVectorizer(stop_words='english')
tfidf=vectorizer.fit_transform(x)
xv=tfidf.toarray()
clf=MultinomialNB()
eq=clf.fit(xv,y)


# Testing for new data
# test='Life is very lovely'
test='i have blue balls'
test_token=word_tokenize(test)
filterd_test=[w for w in test_token if not w in stop_words]
filterd_test=' '.join(filterd_test)
# last_index=len(x)
# filterd_test=pd.Series(filterd_test,index=[last_index])
# x=x.append(filterd_test)
test_tfidf=vectorizer.transform([filterd_test])
test_tfidf_array=test_tfidf.toarray()
# print(test_tfidf_array)
print(eq.predict(test_tfidf_array))