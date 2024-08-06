#importing necessary functions
import pandas as pd
import itertools
import string
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
import matplotlib as plt
import streamlit as st
import  pickle

pd.set_option('display.max_rows',None)

#loading data
df = pd.read_csv('data/drugsComTrain_raw.tsv',sep='\t')

#restructuring the data
df_train = df[(df['condition']=='Birth Control') | (df['condition']=='Depression') | (df['condition']=='High Blood Pressure') | (df['condition']=='Diabetes, Type 2')]
x = df_train.drop(['Unnamed: 0','drugName','rating','date','usefulCount'],axis=1)

#removing the inverted commas in the drug review data
for i,col in enumerate(x.columns):
    x.iloc[:,i] = x.iloc[:,i].str.replace('"','')

pd.set_option('max_colwidth',None)

import nltk
#nltk.download()

#loading stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')

#importing and creating a lemmetizer object
from nltk.stem import WordNetLemmatizer
lemmetizer = WordNetLemmatizer()


from bs4 import BeautifulSoup
import re

#creating a cleaninng function
def review_to_words(raw_review):
    #removing html tags
    review_text = BeautifulSoup(raw_review,'html.parser').get_text()
    #removing any punctuation
    letters_only = re.sub('[^a-zA-Z] ',' ',review_text)
    #lowercasing and tokenising the data
    words = letters_only.lower().split()
    #removing stop words
    meaningful_words=[w for w in words if not w in stop]
    #lemmetizing each word to its base word
    lemmetized_words=[lemmetizer.lemmatize(w) for w in meaningful_words]
    #re-combining the words to sentences
    cleaned_output = " ".join(lemmetized_words)
    return cleaned_output

#applying the cleaning function to the drug reviews in the dataset
x['review_clean'] = x['review'].apply(review_to_words)

#creating the data and labels
x_feat = x['review_clean']
y = x['condition']

#spliting the data to training and data set
x_train,x_test,y_train,y_test = train_test_split(x_feat,y,stratify=y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer

#creating a vectorizer object and converting the training and test data to tfidf values
tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.8,ngram_range=(1,3))
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

#creating a passive aggressive classifier object and training the model with the tfidf training data
pass_tf = PassiveAggressiveClassifier()
pass_tf.fit(tfidf_train,y_train)

#evaluating the performance of the model with the tfidf test data
pred = pass_tf.predict(tfidf_test)
score = metrics.accuracy_score(y_test,pred)
print(f"accuracy: {score}")

#experimenting the model with actual input data
text = ["nexplanon since dec 27 201 got first period end january lasted month half march 201 bleed close three week started bleeding march 28th bleeding every since gained 1 lb far since getting birth control although weight gain deal breaker bleeding is. trying patient see body adjusts implant three month far finger crossed cycle go away awhile."]
test = tfidf_vectorizer.transform(text)
pred = pass_tf.predict(test)[0]
print(pred)

#creating the drug recommendation function
def top_drugs_extractor(condition,df):
    df_top = df[(df['rating']>=9) & (df['usefulCount']>= 100)].sort_values(by =['rating','usefulCount'],ascending=[False,False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst

#experimenting the function
drugs = top_drugs_extractor('Depression',df)
print(drugs)

with open('vectorizer/trained_vect.sav', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('model/trained_model.sav', 'wb') as f:
    pickle.dump(pass_tf, f)
