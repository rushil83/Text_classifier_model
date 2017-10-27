import os
from helper_file.py import load_files, bag_of_words, load_data 
from sklearn_file.py import sklearn_model, loading_file_sklearn, preprocessing_sklearn_model
from numpy_file.py import numpy_classifier_model
from tensorflow_file.py import data_to_tensor, neural_network_framework, train_neural_network

## laoding files in X, Y from our helper_files

path1 = '/home/rushi/Downloads/python/text/file/computing.txt'
path2 = '/home/rushi/Downloads/python/text/file/earth_beneath.txt'

text_file_type1 = load_files(path1) 
text_file_type2 = load_files(path2) 

bag_of_words = bag_of_words(text_file_type1, text_file_type2)
x,y = load_data(text_file_type1,text_file_type2,bag_of_words)



							     ## Sklearn-Classifier Model

## use any of the algorithm in your model, as per accuracy concern

algo1 = sklearn.linear_model.LogisticRegression
algo2 = sklearn.tree.DecisionTreeClassifier
algo3 = sklearn.naive_bayes.GaussianNB
algo4 = sklearn.svm.SVC

model = sklearn_model(algo1,x,y)

 ## we can directly load and preprocess(nlp stuff) our text_files in sklearn,
 ## we have created this nlp preprocessing function in our sklearn_file.py

sklearn_text_file_1 = loading_file_sklearn(path1)
sklearn_text_file_2 = loading_file_sklearn(path2)

preprocessing_sklearn_model = preprocessing_sklearn_model(algo1,sklearn_text_file_1,sklearn_text_file_2)


								  ## using numpy to create a simple neural network classifier model

numpy_model = numpy_classifier_model(x,y)

								 ## using tensorflow to create a deep neural network classifier model
x,y = data_to_tensor(x,y)
nn_framework = neural_network_framework(x)

tensorflow_model = train_neural_network(x,y,nn_framework) 

