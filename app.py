"""
Created on Wed Jun 29 23:10:59 2022

@author: bhavesh
"""
# Import Dependencies
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# NLP Model
df = pd.read_csv(r'spam_dataset.csv', encoding="ISO-8859-1")
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis = 1)
df.rename(columns={'v1':'Labels', 'v2': 'Message'}, inplace=True)
df.drop_duplicates(inplace=True)
df['Labels'] = df['Labels'].map({'ham':0, 'spam':1})

def clean_data(Message):
    Message_without_punc = [ character for character in Message if character not in string.punctuation]
    Message_without_punc = ''.join(Message_without_punc)
    separator = ''
    return separator.join([word for word in Message_without_punc.split() if word.lower() not in stopwords.words('english') ])
    
df['Message'] = df['Message'].apply(clean_data)

x = df['Message']
y = df['Labels']

cv = CountVectorizer()

x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = MultinomialNB().fit(x_train, y_train)

prediction = model.predict(x_test)

#print(accuracy_score(y_test, prediction))
#print(confusion_matrix(y_test, prediction))
#print(classification_report(y_test, prediction))


def predict(text):
    Labels = ['Not Spam', 'Spam']
    x = cv.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str('This message is looking  to be: '+ Labels[v])

## print(predict(['Congratulations you won a lottery ticket']))

st.title('Spam Classifier')
st.image('PB.jpg')    
user_input = st.text_input(' Write your Message')
submit = st.button('Predict')

if submit:
    answer = predict([user_input])
    st.text(answer)
    
    
    



