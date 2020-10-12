# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:13:00 2020

@author: Frederick
"""

import numpy as np
import pandas as pd

df = pd.read_csv('mining_headlines_500.csv', sep=',')

#Check for nulls
print(df.isnull().sum())

# Check for whitespace strings (it's OK if there aren't any!):
blanks = []  # start with an empty list

for i,lb,rv in df.itertuples():  # iterate over the DataFrame
    if type(rv)==str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list
        
print('Blanks:', len(blanks))

df.dropna(inplace=True)

df['label'].value_counts()

from sklearn.model_selection import train_test_split

X = df['headline']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

stopwords = ['a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', 
             'even', 'ever', 'for', 'from', 'get', 'had', 'has', 'have', 'he', 'her', 'hers', 'his', 
             'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me', 'my', 'of', 'on', 'or', 
             'see', 'seen', 'she', 'so', 'than', 'that', 'the', 'their', 'there', 'they', 'this', 
             'to', 'was', 'we', 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'you']

# Model Selection
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

clf = LinearSVC()
#stop_words=stopwords

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', clf),
])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  

# Form a prediction set
predictions = text_clf.predict(X_test)

# Report the confusion matrix
from sklearn import metrics

#print(metrics.confusion_matrix(y_test,predictions))

# You can make the confusion matrix less confusing by adding labels:
dfc = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['neg','pos'], columns=['neg','pos'])

# Print a classification report
print(metrics.classification_report(y_test,predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))