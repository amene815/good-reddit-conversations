from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import csv

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

import networkx


import pandas as pd
data1 = pd.read_csv('2017-11.csv')
data2 = pd.read_csv('2017-12.csv')
data3 = pd.read_csv('2018-01.csv')
data4 = pd.read_csv('2018-02.csv')
data5 = pd.read_csv('2018-03.csv')

data = pd.concat([data1,data2,data3,data4,data5])
print("Data size = {}".format(len(data)))

# Drop N/A Values
data = data.dropna(subset=['body'],inplace=True)

# vectorizer = CountVectorizer()
# piece_of_data = vectorizer.fit_transform(data['body'])
# vocab = vectorizer.vocabulary_
# print("Vocab size = {}".format(len(vocab)))

# Pre Processing

#To Lower Case
data['body'] = data['body'].str.lower()


# Tokenize
print('\nTokenize')
def identify_tokens(row):
    comment = row['body']
    tokens = nltk.word_tokenize(comment)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

data['words'] = data.apply(identify_tokens,axis=1)

# Stemming 
print('\nStemming')
from nltk.stem import PorterStemmer
stemming = PorterStemmer()

def stem_list(row):
    my_list = row['words']
    stemmed_list = [stemming.stem(word) for word in my_list]
    return (stemmed_list)

data['stemmed_words'] = data.apply(stem_list,axis=1)

# Remove Stopwords
print('\nStopwords')
stop_words = set(stopwords.words('english'))

def remove_stops(row):
    my_list = row['stemmed_words']
    meaningful_words = [w for w in my_list if not w in stop_words]
    return (meaningful_words)

data['stem_meaningful'] = data.apply(remove_stops,axis=1)

print(data['stem_meaningful'][0])

# Linear Regression
# Use scikit


