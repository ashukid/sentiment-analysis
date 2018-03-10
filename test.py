import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize

with open("all.txt") as file:
    default=file.readlines()

# print(default[1])

data=[]
for s in default:
	if(s !="\r\n" and s !="\n"):
		data.append(list(map(str,s.split(':'))))

# for i in range(len(data)):
# 	print(i)
# 	print(data[i])

sentiment=[]
document=[]
for i in range(len(data)):
	sentiment.append(data[i][0])
	document.append(data[i][1].decode('ISO-8859-1'))

# print(sentiment)

train=pd.DataFrame({'Sentiment':sentiment,'Document':document})
train.loc[train["Sentiment"]=='positive','Sentiment']=1
train.loc[train["Sentiment"]=='negative','Sentiment']=0
train.loc[train["Sentiment"]=='neutral','Sentiment']=2

x=train['Document']
y=train['Sentiment'].astype(int)

for i in range(len(x)):
	sentence=x.iloc[i]
	sentence=word_tokenize(sentence)
	x.iloc[i]=" ".join(sentence)

vectorizer=TfidfVectorizer()
tfidf=vectorizer.fit_transform(x)
xv=tfidf.toarray()
clf=MultinomialNB()
eq=clf.fit(xv,y)

# testing

with open("test.txt") as file:
    default=file.readlines()


testdata=[]
for s in default:
	if(s !="\r\n" and s !="\n"):
		testdata.append(list(map(str,s.split(':'))))


sentiment=[]
document=[]
for i in range(len(testdata)):
	sentiment.append(testdata[i][0])
	document.append(testdata[i][1].decode('ISO-8859-1'))

test=pd.DataFrame({'Sentiment':sentiment,'Document':document})
test.loc[test["Sentiment"]=='positive','Sentiment']=1
test.loc[test["Sentiment"]=='negative','Sentiment']=0
test.loc[test["Sentiment"]=='neutral','Sentiment']=2

xtest=test['Document']
ytest=test['Sentiment'].astype(int)
ytest=np.ravel(ytest)

for i in range(len(xtest)):
	sentence=xtest.iloc[i]
	sentence=word_tokenize(sentence)
	xtest.iloc[i]=" ".join(sentence)

testtfidf=vectorizer.transform(xtest)
testtfidf_array=testtfidf.toarray()
ypredict=eq.predict(testtfidf_array)
print("Predicted output : {}".format(ypredict))
print("Original output : {}".format(ytest))

count=0
for i in range(len(ytest)):
	if(ytest[i]==ypredict[i]):
		count+=1
print("Accuracy : {}%".format(float(count)/len(ytest)*100))


