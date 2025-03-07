#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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




#########################################################
### your code goes here ###
print(len(features_train[0]))

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"


from sklearn.metrics import accuracy_score
t1 = time()
#prediction = clf.predict(features_test)
print(accuracy_score(clf.predict(features_test), labels_test))
print "predict time:", round(time()-t1, 3), "s"

#filtered = filter(lambda x: x == 1, prediction)
#print "Chris Emails: ", len(filtered)

#########################################################


