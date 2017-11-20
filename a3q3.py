from __future__ import print_function
import numpy as np
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import ssl
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.examples.tutorials.mnist import input_data

# This restores the same behavior as before.
# need python imageing library to run PIL to run 
# command: python3 -m pip install Pillow

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

n_components = 900

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


inputsORth=X_train_pca
inputsOrig=X_train
trainYs=y_train
testsOrig=X_test
testsORth=X_test_pca
testYs=y_test



#with original values
trX, trY, teX, teY = inputsOrig, trainYs, testsOrig, testYs
setOfX=[]
setOfY=[]

setOfX=trX
setOfY=trY

kf = KFold(n_splits=4, shuffle=True);
print("Results of NN using original data as input: ")
index=0
for train_index, test_index in kf.split(setOfX):
    X_train, X_test = setOfX[train_index], setOfX[test_index]
    y_train, y_test = setOfY[train_index], setOfY[test_index]
    print("Done training for Itteration")
    index=index+1
    print(index)
    
    mlp = MLPClassifier(hidden_layer_sizes=(7,7,7),max_iter=500)
    mlp.fit(X_train,y_train)


    predictions = mlp.predict(X_test)

    print(classification_report(y_test,predictions))

print("Results not using k means ")
mlp = MLPClassifier(hidden_layer_sizes=(7,7,7),max_iter=500)
mlp.fit(trX,trY)

predictions = mlp.predict(teX)

print(classification_report(teY,predictions))



#with orthogonal values
print("Results of NN using orthogonal basis as input: ")
trX, trY, teX, teY = inputsORth, trainYs, testsORth, testYs
setOfX=[]
setOfY=[]
setOfX=trX
setOfY=trY

kf = KFold(n_splits=4, shuffle=True);
index=0
for train_index, test_index in kf.split(setOfX):
    X_train, X_test = setOfX[train_index], setOfX[test_index]
    y_train, y_test = setOfY[train_index], setOfY[test_index]
    print("Done training for Itteration")
    index=index+1
    print(index)
    
    mlp = MLPClassifier(hidden_layer_sizes=(7,7,7),max_iter=500)
    mlp.fit(X_train,y_train)

    predictions = mlp.predict(X_test)

    print(classification_report(y_test,predictions))



print("Results not using k means ")

mlp = MLPClassifier(hidden_layer_sizes=(7,7,7),max_iter=500)
mlp.fit(trX,trY)

predictions = mlp.predict(teX)

print(classification_report(teY,predictions))





