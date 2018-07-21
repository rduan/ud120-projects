#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]




#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score
from collections import Counter

# clf=svm.SVC(kernel='linear')
# clf=svm.SVC(kernel='rbf', C=10)
# clf.fit(features_train,labels_train)
# pre10 = clf.predict(features_test)
# print(accuracy_score(labels_test, pre10))

# clf=svm.SVC(kernel='rbf', C=100)
# clf.fit(features_train,labels_train)
# pre100 = clf.predict(features_test)
# print(accuracy_score(labels_test, pre100))

# clf=svm.SVC(kernel='rbf', C=1000)
# clf.fit(features_train,labels_train)
# pre1000 = clf.predict(features_test)
# print(accuracy_score(labels_test, pre1000))

# clf=svm.SVC(kernel='rbf', C=1000
# clf.fit(features_train,labels_train)
# pre10000 = clf.predict(features_test)
# print(accuracy_score(labels_test, pre10000))

clf=svm.SVC(kernel='rbf', C=10000)
clf.fit(features_train,labels_train)
pre10000 = clf.predict(features_test)

# print(accuracy_score(labels_test, pre10000))
#########################################################

# print(pre10000[10])
# print(pre10000[26])
# print(pre10000[50])

print(Counter(pre10000))