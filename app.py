"""
Created on Wed Jun 29 23:10:59 2022

@author: bhavesh
"""

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

# NLP Model
df = pd.read_csv(r'spam_dataset.csv', encoding="ISO-8859-1")
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis = 1)
df.rename(columns={'v1':'Labels', 'v2': 'Message'}, inplace=True)
df.drop_duplicates(inplace=True)
df['Labels'] = df['Labels'].map({'ham':0, 'spam':1})
print(df.head())

def clean_data(Message):
    Message_without_punc = [ character for character in Message if character not in string.punctuation]
    Message_without_punc = ''.join(Message_without_punc)
    separator = ''
    return separator.join([word for word in Message_without_punc.split() if word.lower() not in stopwords.words('english') ])
