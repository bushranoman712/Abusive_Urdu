# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:20:44 2019

@author: Noman
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:45:52 2019

@author: Noman
"""


import numpy as np
import pandas as pd
# Import adjustText, initialize list of texts
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import itertools
import os
import csv
import re

#Seed Random Numbers with the TensorFlow Backend
from numpy.random import seed
seed(1)
import natsort 



class Urdu_Dataset:

    def avg_word(sentence):
      words = sentence.split()
      return (sum(len(word) for word in words)/len(words))
  
    def Generate_Urdu_Ngrams(self, _ngram_range=(1,1), _max_features=5000, words = True ):    
        
        
        dataset_path = "F:\Machine_Learning\\Basic_Models\\maaz_data_set\\abusive_language_recognation_dataset.csv"
    
        #load data
        df = pd.read_csv(dataset_path, usecols=['Tweets', 'Positive', 'Negative'])
        print (df.head())
        #print(df.shape)
        rows,col = df.shape

        df['word_count'] = df['Tweets'].apply(lambda x: len(str(x).split(" ")))
        df[['Tweets','word_count']].head()
        
        df['char_count'] = df['Tweets'].str.len() ## this also includes spaces
        df[['Tweets','char_count']].head()
        
        
        df['avg_word'] = df['Tweets'].apply(lambda x:  Urdu_Dataset.avg_word(x))
        df[['Tweets','avg_word']].head()
        
        
        print(df.describe())
        print(df.sum(axis=0) )
        
        
        yeswisedf = df[(df['Positive'] == 'yes')]
        print(yeswisedf.head())
        
        
        
        yeswisedf['word_count'] = yeswisedf['Tweets'].apply(lambda x: len(str(x).split(" ")))
        yeswisedf[['Tweets','word_count']].head()
        
        yeswisedf['char_count'] = yeswisedf['Tweets'].str.len() ## this also includes spaces
        yeswisedf[['Tweets','char_count']].head()
        
        
        yeswisedf['avg_word'] = yeswisedf['Tweets'].apply(lambda x:  Urdu_Dataset.avg_word(x))
        yeswisedf[['Tweets','avg_word']].head()
        
        
        print(yeswisedf.describe())
        print(yeswisedf.sum(axis=0) )
        
        
        
        nowisedf = df.loc[df['Negative'] == 'no']
        print(nowisedf.head())
        
        
        
        nowisedf['word_count'] = nowisedf['Tweets'].apply(lambda x: len(str(x).split(" ")))
        nowisedf[['Tweets','word_count']].head()
        
        nowisedf['char_count'] = nowisedf['Tweets'].str.len() ## this also includes spaces
        nowisedf[['Tweets','char_count']].head()
        
        
        nowisedf['avg_word'] = nowisedf['Tweets'].apply(lambda x:  Urdu_Dataset.avg_word(x))
        nowisedf[['Tweets','avg_word']].head()
        
        
        print(nowisedf.describe())
        print(nowisedf.sum(axis=0) )
        
        
        #nowisedf = nowisedf[:3000]
        
        
        df_row = pd.concat([yeswisedf['Tweets'], nowisedf['Tweets']])
        print(df_row.head())
        
        
        print("Total Dataset:" + str(len(df_row)))
        print("yes class:" + str(len(yeswisedf)))
        print("no class:" + str(len(nowisedf)))
        
        
        
        
        
        Number_OF_Documents = rows + 1
        #Number_OF_Documents = len(yeswisedf) + len(nowisedf)
        Number_OF_POSITIVE_SAMPLES = len(yeswisedf)
        Number_OF_NEGATIVE_SAMPLES = len(nowisedf)
        
        
        
        y_train = np.empty(Number_OF_Documents-1)
       
        for i in range(0,Number_OF_Documents-1):
            if i < Number_OF_POSITIVE_SAMPLES - 1:
                y_train[i] = 1
            else:
                y_train[i] = 0
        
        
        i = 0
        #for index,row in df[:Number_OF_Documents].iterrows():
        for index,row in df.iterrows():
            positive = row['Positive']
            negative = row['Negative']
        
            
            if positive == "yes":
                y_train[i] = 1
            elif negative == "no":
                y_train[i] = 0
            else:
                y_train[i] = 0
            i = i + 1
            
            
        
        
        keywords_dictionary = []
        sentences_corpus = []
        #for index,row in df.iterrows(): 
        for index,row in df[:Number_OF_Documents].iterrows():
                text = str(row['Tweets'])
                
                sentences_corpus.append(text)
                list_of_words = text.split(" ")
                keywords_dictionary.append(list_of_words)
            
        
        
        print(keywords_dictionary[0])
        #print(sentences_corpus[1])
        
        
        vocab = []
        for kl in keywords_dictionary: 
            for w in kl:
                vocab.append(str(w))
            
        
        
        
        corpus = []
        for kl in sentences_corpus: 
            corpus = corpus + kl.split(',')
        
        
    
        if words:
            vectorizer = CountVectorizer(ngram_range=_ngram_range, max_features=_max_features)
            Count_Vect = vectorizer.fit_transform(vocab)
            #print(vectorizer.get_feature_names())
            #print(X.toarray())
        else:
            vectorizer = CountVectorizer(ngram_range=_ngram_range, token_pattern = r"(?u)\b\w+\b",  analyzer='char')
            Count_Vect = vectorizer.fit_transform(vocab)
            #print(vectorizer.get_feature_names())

                
                
        vectorizer = TfidfVectorizer(ngram_range=_ngram_range,max_features=_max_features) # You can still specify n-grams here.
        X = vectorizer.fit_transform(corpus).toarray()   
            
            
            
        print( "Shape of final Ngram vector:" + str(X.shape))
        print( "Shape of labels:" + str(y_train.shape))
        xTrain, xTest, yTrain, yTest = train_test_split(X, y_train, test_size = 0.2)   
        return xTrain, xTest, yTrain, yTest, sentences_corpus, keywords_dictionary, y_train
    



