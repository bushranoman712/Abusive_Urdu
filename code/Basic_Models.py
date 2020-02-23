# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:40:06 2019

@author: Noman
"""

import warnings
warnings.filterwarnings("ignore")
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy
import numpy as np
from numpy.random import seed
seed(1)
import pandas as pd

from keras.preprocessing import sequence
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention

from keras import backend
from keras import backend as K
from keras import models
from keras.models import Sequential
from keras.layers import Dense,LSTM, Dropout, Flatten, Embedding, Bidirectional, GlobalAveragePooling1D
from keras.layers import Conv1D,Conv2D, MaxPooling2D, MaxPooling1D, GlobalMaxPooling1D

from keras.optimizers import SGD,RMSprop,Adam
from keras.regularizers import l2,l1

from keras.callbacks import CSVLogger
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score,fbeta_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

import numpy as np
from numpy.random import seed
seed(1)

class BasicModels(object):
    
    
    
    def Shape_Reshaper(_xTrain, _xTest):    
        xTrain_arr = _xTrain
        xTest_arr = _xTest
    
        
        dim1,dim3 = _xTrain.shape
        dim2 = 1        
        
        xTrain = np.reshape(_xTrain, (dim1, dim2 , dim3)) #xTrain_arr.reshape(dim1, dim2 , dim3)
        print(xTrain.shape)       
        
        t_dim1,t_dim3 = _xTest.shape
        t_dim2 = 1
        xTest = xTest_arr.reshape(t_dim1, t_dim2 , t_dim3)
        print(xTest.shape)
        
        _input_shape = (dim2,dim3)
        print(_input_shape)
       
        
        return xTrain,xTest,_input_shape
    
    
    def SVM_Ngrams(self,xTrain, xTest, yTrain, yTest, _C = [1], _kernel= ['linear', 'rbf'], _tol = [0.01, 0.02], _class_weight = 'balanced'):
        print("--------------- Support Vector Machine ---------------")
        print("Fitting the classifier to the training set")
        param_grid = {'C': _C, 'kernel': _kernel, 'tol': _tol}
        clf = GridSearchCV(svm.SVC(probability=True,class_weight='balanced'), param_grid)
        clf.fit(xTrain, yTrain)
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        y_pred = clf.predict(xTest)
        print("=== AUC Score ===")
        accuracy = accuracy_score(yTest, y_pred)
        print('Accuracy: %f' % accuracy)
        
        
        print("=== Classification Report SVM ===")
        print(classification_report(yTest, y_pred))
        print('\n')
        print("=== Confusion Matrix SVM ===")
        print(confusion_matrix(yTest, y_pred))
        print('\n')
        print("--------------- Support Vector Machine ---------------")
        return clf

        
        
    def RF_Ngrams(self,xTrain, xTest, yTrain, yTest, _n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=10):
        print("--------------- Random Forest ---------------")
       
        
        rfc = RandomForestClassifier(n_estimators = _n_estimators, max_features=None,  n_jobs=_n_jobs, criterion = _criterion, max_depth  = _max_depth, verbose=_verbose)

        
        rfc = rfc.fit(xTrain, yTrain)
        
        print("Predicting labels for test data..")
        rfc_predict  = rfc.predict(xTest)
        
        print("=== Classification Report Random Forest ===")
        print(classification_report(yTest, rfc_predict))
        print('\n')
        print("=== Confusion Matrix Random Forest ===")
        print(confusion_matrix(yTest, rfc_predict))
        print('\n')
        print("--------------- Random Forest ---------------")
        return rfc
    
    

    def CNN1D_Ngrams(self,_xTrain, _xTest, yTrain, yTest, _loss='mean_squared_error', _optimizer= 'Adam', _metrics=['accuracy'], _epochs = 25 , _validation_split = 0.2, _batch_size = 4, _verbose = 2 ):
        print("--------------- CNN1D ---------------")  
        
        
        
        xTrain,xTest,_input_shape = BasicModels.Shape_Reshaper(_xTrain, _xTest)
        
        model = Sequential()
        model.add(Conv1D(32, (3),strides=3, padding='same',activation='tanh', input_shape=_input_shape))
        model.add(Dense(32 ,activation='tanh'))
        model.add(Dropout(0.1))
        model.add(Dense(64 ,activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Flatten()) 
        model.add(Dense(1 ,activation='sigmoid'))
        model.summary()
        model.compile(loss=_loss, optimizer=_optimizer, metrics=_metrics)
        
        model.fit(xTrain, yTrain, epochs=_epochs,  validation_split=_validation_split)
       
        accuracy = model.evaluate(xTest, yTest,  verbose=_verbose)
        print(accuracy)
        y_pred = model.predict_classes(xTest)

        print("=== Classification Report Conv 1D ===")
        print(classification_report(yTest, y_pred))
        print('\n')

        print("=== Confusion Matrix Conv 1D ===")
        print(confusion_matrix(yTest, y_pred))
        print('\n')
                
        print("--------------- CNN1D ---------------") 
        return model
    
 
    def bc_LSTM_Ngrams(self,_xTrain, _xTest, yTrain, yTest, _loss='mean_squared_error', _optimizer= 'SGD', _metrics=['accuracy'], _epochs = 25 , _validation_split = 0.2, _batch_size = 2, _verbose = 0 ):
        print("--------------- LSTM ---------------")       
        
        xTrain,xTest,_input_shape = BasicModels.Shape_Reshaper(_xTrain, _xTest)
        
        model = Sequential()
        model.add(LSTM(4, input_shape=_input_shape, activation='tanh' , return_sequences=True))
        model.add(Dropout(0.5))
        model.add(Flatten()) 
        model.add(Dense(1 ,activation='sigmoid'))
        model.summary()
        model.compile(loss=_loss, optimizer= _optimizer, metrics=_metrics)
        history = model.fit(xTrain, yTrain, epochs=_epochs,  validation_split = _validation_split)
        
        accuracy = model.evaluate(xTest, yTest,  verbose= _verbose)
        y_pred = model.predict_classes(xTest, batch_size = _batch_size, verbose = _verbose)
        
        
        print("=== Classification Report LSTM ===")
        print(classification_report(yTest, y_pred))
        print('\n')
        print("=== Confusion Matrix LSTM ===")
        print(confusion_matrix(yTest, y_pred))
        print('\n')
        print("--------------- LSTM ---------------")
        return model



   
