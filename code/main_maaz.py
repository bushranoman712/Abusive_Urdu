# -*- coding: utf-8 -*-

"""
Created on Sat Oct 12 12:47:21 2019

@author: Noman
"""
import Basic_Models as bm
basic_model = bm.BasicModels()

#urdu dataset mazz paper
import maaz_keywords_generate as ud
dsurdu = ud.Urdu_Dataset()



import pandas as pd
from gensim import models
from sklearn.model_selection import train_test_split

from gensim.models.wrappers import FastText
from gensim.models.fasttext import FastText, load_facebook_vectors
from gensim.models.keyedvectors import KeyedVectors


import numpy
import numpy as np
from numpy.random import seed
seed(1)


#%% fast load
wxTrain1, wxTest1, wyTrain1, wyTest1, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(_ngram_range=(1,1), _max_features=3000, words= True)

w2v_file_fast_text = "C:\FasText\\cc.ur.300.bin.gz"
w2vmodel = FastText.load_fasttext_format(w2v_file_fast_text)   
print("Word 2 Vector File Loaded!")    

    
vector = w2vmodel.wv['easy']
print( "Shape of Vector:" + str(vector.shape))
        

#%% fast vector generation
vectors_per_document = 5  
X_train_Vector = []
for kl in keywords_dictionary:
    vector_list = []
    for word in kl[0:vectors_per_document]:
        if word in w2vmodel.wv.vocab:
            vector_list.append(w2vmodel[word])
        else:
            vector_list.append(np.random.uniform(-0.1, 0.1, 300))
            
    matrix_2d = np.array(vector_list)
                #print(matrix_2d.shape)
    average_sentence_vector = np.mean(matrix_2d, axis = 0)
            
    X_train_Vector.append(average_sentence_vector)
        
X = numpy.array(X_train_Vector)
print( "Shape of training documents" + str(X.shape)) 



#%% fast algorithms

import Basic_Models as bm
basic_model = bm.BasicModels()  
fxTrain, fxTest, fyTrain, fyTest = train_test_split(X, labels, test_size = 0.2)
print(fxTrain.shape)

frf_ngram = basic_model.RF_Ngrams(fxTrain, fxTest, fyTrain, fyTest,_n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=2)
fsvm_ngram = basic_model.SVM_Ngrams(fxTrain, fxTest, fyTrain, fyTest, _C = [1], _kernel= ['linear'], _tol = [0.01], _class_weight = 'balanced')

   
fcnn1d_ngram = basic_model.CNN1D_Ngrams(fxTrain, fxTest, fyTrain, fyTest,_epochs = 50, _verbose=2)
flstm_ngram = basic_model.bc_LSTM_Ngrams(fxTrain, fxTest, fyTrain, fyTest,_epochs = 50, _verbose=5)



#%% fast roc
models = [frf_ngram,fsvm_ngram, fcnn1d_ngram,flstm_ngram]
xTest_data = [fxTest,fxTest,fxTest,fxTest]
yTest_data = [fyTest,fyTest,fyTest,fyTest]
labels = ["fasttext-w2v-RF", "fasttext-w2v-SVM","fasttext-w2v-1DCNN",  "fasttext-w2v-LSTM"]
basic_model.ROC_CURVE_ALL_MODEL_Ngram(xTest_data, yTest_data, 2, models,labels, _linestyle = ':', _figsize = (8,8))





#%% char cell
#unigram char
xTrain1, xTest1, yTrain1, yTest1, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(_ngram_range=(1,1), _max_features=3000, words= False)



rf_ngram1 = basic_model.RF_Ngrams(xTrain1, xTest1, yTrain1, yTest1,_n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=2)
svm_ngram1 = basic_model.SVM_Ngrams(xTrain1, xTest1, yTrain1, yTest1, _C = [1], _kernel= ['linear'], _tol = [0.01], _class_weight = 'balanced')  
cnn1d_ngram1  = basic_model.CNN1D_Ngrams(xTrain1, xTest1, yTrain1, yTest1,_epochs = 50, _verbose=2)
lstm_ngram1 = basic_model.bc_LSTM_Ngrams(xTrain1, xTest1, yTrain1, yTest1,_epochs = 50, _verbose=2)


#bigram char
xTrain2, xTest2, yTrain2, yTest2, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(_ngram_range=(2,2), _max_features=3000, words= False)
rf_ngram2 = basic_model.RF_Ngrams(xTrain2, xTest2, yTrain2, yTest2,_n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=2)
svm_ngram2 = basic_model.SVM_Ngrams(xTrain2, xTest2, yTrain2, yTest2, _C = [1,5], _kernel= ['linear','rbf'], _tol = [0.01,0.02], _class_weight = 'balanced')  
cnn1d_ngram2  = basic_model.CNN1D_Ngrams(xTrain2, xTest2, yTrain2, yTest2,_epochs = 50, _verbose=2)
lstm_ngram2 = basic_model.bc_LSTM_Ngrams(xTrain2, xTest2, yTrain2, yTest2,_epochs = 50, _verbose=2)



#trigram char
xTrain3, xTest3, yTrain3, yTest3, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(_ngram_range=(3,3), _max_features=3000, words= False)
rf_ngram3 = basic_model.RF_Ngrams(xTrain3, xTest3, yTrain3, yTest3,_n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=2)
svm_ngram3 = basic_model.SVM_Ngrams(xTrain3, xTest3, yTrain3, yTest3, _C = [1], _kernel= ['linear'], _tol = [0.01], _class_weight = 'balanced')  
cnn1d_ngram3  = basic_model.CNN1D_Ngrams(xTrain3, xTest3, yTrain3, yTest3,_epochs = 50, _verbose=2)
lstm_ngram3 = basic_model.bc_LSTM_Ngrams(xTrain3, xTest3, yTrain3, yTest3,_epochs = 50, _verbose=2)




#uni+bigram char
xTrain12, xTest12, yTrain12, yTest12, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(_ngram_range=(1,2), _max_features=3000, words= False)
rf_ngram12 = basic_model.RF_Ngrams(xTrain12, xTest12, yTrain12, yTest12,_n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=2)
svm_ngram12 = basic_model.SVM_Ngrams(xTrain12, xTest12, yTrain12, yTest12, _C = [1], _kernel= ['linear'], _tol = [0.01], _class_weight = 'balanced')  
cnn1d_ngram12  = basic_model.CNN1D_Ngrams(xTrain12, xTest12, yTrain12, yTest12,_epochs = 50, _verbose=2)
lstm_ngram12 = basic_model.bc_LSTM_Ngrams(xTrain12, xTest12, yTrain12, yTest12,_epochs = 50, _verbose=2)



#uni+trigram char
xTrain13, xTest13, yTrain13, yTest13, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(_ngram_range=(1,3), _max_features=3000, words= False)
rf_ngram13 = basic_model.RF_Ngrams(xTrain13, xTest13, yTrain13, yTest13,_n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=2)
svm_ngram13 = basic_model.SVM_Ngrams(xTrain13, xTest13, yTrain13, yTest13, _C = [1], _kernel= ['linear'], _tol = [0.01], _class_weight = 'balanced')  
cnn1d_ngram13  = basic_model.CNN1D_Ngrams(xTrain13, xTest13, yTrain13, yTest13,_epochs = 50, _verbose=2)
lstm_ngram13 = basic_model.bc_LSTM_Ngrams(xTrain13, xTest13, yTrain13, yTest13,_epochs = 50, _verbose=2)





#unigram word
wxTrain1, wxTest1, wyTrain1, wyTest1, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(_ngram_range=(1,1), _max_features=3000, words= True)

wrf_ngram1 = basic_model.RF_Ngrams(wxTrain1, wxTest1, wyTrain1, wyTest1,_n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=2)
wsvm_ngram1 = basic_model.SVM_Ngrams(wxTrain1, wxTest1, wyTrain1, wyTest1, _C = [1], _kernel= ['linear'], _tol = [0.01], _class_weight = 'balanced')  
wcnn1d_ngram1  = basic_model.CNN1D_Ngrams(wxTrain1, wxTest1, wyTrain1, wyTest1,_epochs = 50, _verbose=2)
wlstm_ngram1 = basic_model.bc_LSTM_Ngrams(wxTrain1, wxTest1, wyTrain1, wyTest1,_epochs = 50, _verbose=2)



#bigram word
wxTrain2, wxTest2, wyTrain2, wyTest2, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(_ngram_range=(2,2), _max_features=3000, words= True)

wrf_ngram2 = basic_model.RF_Ngrams(wxTrain2, wxTest2, wyTrain2, wyTest2,_n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=2)
wsvm_ngram2 = basic_model.SVM_Ngrams(wxTrain2, wxTest2, wyTrain2, wyTest2, _C = [1,5], _kernel= ['linear','rbf'], _tol = [0.01,0.02], _class_weight = 'balanced')  
wcnn1d_ngram2  = basic_model.CNN1D_Ngrams(wxTrain2, wxTest2, wyTrain2, wyTest2,_epochs = 50, _verbose=2)
wlstm_ngram2 = basic_model.bc_LSTM_Ngrams(wxTrain2, wxTest2, wyTrain2, wyTest2,_epochs = 50, _verbose=2)




#trigram word
wxTrain3, wxTest3, wyTrain3, wyTest3, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(_ngram_range=(3,3), _max_features=3000, words= True)

wrf_ngram3 = basic_model.RF_Ngrams(wxTrain3, wxTest3, wyTrain3, wyTest3, _n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=2)
wsvm_ngram3 = basic_model.SVM_Ngrams(wxTrain3, wxTest3, wyTrain3, wyTest3, _C = [1], _kernel= ['linear'], _tol = [0.01], _class_weight = 'balanced')  
wcnn1d_ngram3  = basic_model.CNN1D_Ngrams(wxTrain3, wxTest3, wyTrain3, wyTest3, _epochs = 50, _verbose=2)
wlstm_ngram3 = basic_model.bc_LSTM_Ngrams(wxTrain3, wxTest3, wyTrain3, wyTest3, _epochs = 50, _verbose=2)




#uni+bigram word
wxTrain12, wxTest12, wyTrain12, wyTest12, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(_ngram_range=(1,2), _max_features=3000, words= True)

wrf_ngram12 = basic_model.RF_Ngrams(wxTrain12, wxTest12, wyTrain12, wyTest12, _n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=2)
wsvm_ngram12 = basic_model.SVM_Ngrams(wxTrain12, wxTest12, wyTrain12, wyTest12, _C = [1], _kernel= ['linear'], _tol = [0.01], _class_weight = 'balanced')  
wcnn1d_ngram12  = basic_model.CNN1D_Ngrams(wxTrain12, wxTest12, wyTrain12, wyTest12, _epochs = 50, _verbose=2)
wlstm_ngram12 = basic_model.bc_LSTM_Ngrams(wxTrain12, wxTest12, wyTrain12, wyTest12, _epochs = 50, _verbose=2)



#uni+trigram word
wxTrain13, wxTest13, wyTrain13, wyTest13, sentences_corpus, keywords_dictionary, labels = dsurdu.Generate_Urdu_Ngrams(_ngram_range=(1,3), _max_features=3000, words= True)

wrf_ngram13 = basic_model.RF_Ngrams(wxTrain13, wxTest13, wyTrain13, wyTest13,_n_estimators = 100, _n_jobs=3, _criterion = 'gini', _max_depth  = 15, _verbose=2)
wsvm_ngram13 = basic_model.SVM_Ngrams(wxTrain13, wxTest13, wyTrain13, wyTest13, _C = [1], _kernel= ['linear','rbf'], _tol = [0.01], _class_weight = 'balanced')  
wcnn1d_ngram13  = basic_model.CNN1D_Ngrams(wxTrain13, wxTest13, wyTrain13, wyTest13, _epochs = 50, _verbose=2)
wlstm_ngram13 = basic_model.bc_LSTM_Ngrams(wxTrain13, wxTest13, wyTrain13, wyTest13, _epochs = 50, _verbose=2)


#character best model
models = [svm_ngram1,svm_ngram12, svm_ngram13,svm_ngram2,svm_ngram3]
xTest_data = [xTest1,xTest12,xTest13,xTest2,xTest3]
yTest_data = [yTest1,yTest12,yTest13,yTest2,yTest3]
labels = ["Char-uni-SVM", "Char-uni+bi-SVM","Char-uni+tri-SVM",  "Char-bi-SVM","Char-tri-SVM"]
basic_model.ROC_CURVE_ALL_MODEL_Ngram(xTest_data, yTest_data, models,labels, _linestyle = ':', _figsize = (8,8))

#word best model
ng_to_vec = 6
models = [wsvm_ngram13, wsvm_ngram1,wsvm_ngram12,wsvm_ngram2,wsvm_ngram3]
labels = ["Word-uni+tri-SVM","Word-uni-SVM","Word-uni+bi-SVM", "Word-bi-SVM","Word-tri-SVM"]
xTest_data = [wxTest13,wxTest1,wxTest12,wxTest2,wxTest3]
yTest_data = [wyTest13,wyTest1,wyTest12,wyTest2,wyTest3]
basic_model.ROC_CURVE_ALL_MODEL_Ngram(xTest_data, yTest_data, models,labels, _linestyle = ':', _figsize = (8,8))




#all word and char curve
models = [wsvm_ngram13, wsvm_ngram1,wsvm_ngram12,wsvm_ngram2,wsvm_ngram3,svm_ngram1,svm_ngram12, svm_ngram13,svm_ngram2,svm_ngram3]
labels = ["Word-uni+tri-SVM","Word-uni-SVM","Word-uni+bi-SVM", "Word-bi-SVM","Word-tri-SVM","Char-uni-SVM", "Char-uni+bi-SVM","Char-uni+tri-SVM",  "Char-bi-SVM","Char-tri-SVM"]
xTest_data = [wxTest13,wxTest1,wxTest12,wxTest2,wxTest3,xTest1,xTest12,xTest13,xTest2,xTest3]
yTest_data = [wyTest13,wyTest1,wyTest12,wyTest2,wyTest3,yTest1,yTest12,yTest13,yTest2,yTest3]
basic_model.ROC_CURVE_ALL_MODEL_Ngram(xTest_data, yTest_data, models,labels, _linestyle = ':', _figsize = (8,8))




#best 3 models
ng_to_vec = 6
models = [wsvm_ngram13,svm_ngram1,fsvm_ngram]
labels = ["Word-uni+tri-SVM", "Char-uni-SVM", "Fasttext-w2v-SVM"]
xTest_data = [wxTest13,xTest1,fxTest]
yTest_data = [wyTest13,yTest1,fyTest]
basic_model.ROC_CURVE_ALL_MODEL_Ngram(xTest_data, yTest_data, models,labels, _linestyle = ':', _figsize = (8,8))



