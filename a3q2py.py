import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA
from sklearn import svm
import math
import random
import time
import matplotlib.pyplot as plt
import random as ran

#this som is based on the som from https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
#it has been modified to fit our problem 
class SOM(object):

    def __init__(self, m, n, dim, n_iterations=100):

        # Assign required variables first
        self.m = m; 
        self.n = n
        alpha = 0.2
        sigma = max(m, n) / 2.0
        self.n_iterations = abs(int(n_iterations))

        self.graph = tf.Graph()
        with self.graph.as_default():
            #inits
            # To save data, create weight vectors and their location vectors
            self.weights = tf.Variable(tf.random_normal( [m * n, dim]) )
            self.locations = tf.constant(np.array(list(self.createLocations(m, n))))
            # Training inputs
            self.vect_input = tf.placeholder("float", [dim])
            self.iter_input = tf.placeholder("float")
            #calculations
            #Gets the winner index 
            winnerIndex = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self.weights, tf.stack(
                    [self.vect_input for _ in range(m * n)])), 2), 1)), 0) 
                    
            #gets the winner from the index
            slice_input = tf.pad(tf.reshape(winnerIndex, [1]), np.array([[0, 1]]))
            winnerLocation = tf.reshape(tf.slice(self.locations, slice_input, 
                        tf.constant(np.array([1, 2]), dtype=tf.int64) ), [2])

            # compute new alpha and sigma for each iteration
            learning_rate_op = tf.subtract(1.0, tf.div(self.iter_input, self.n_iterations))
            alpha_op = tf.multiply(alpha, learning_rate_op)
            sigma_op = tf.multiply(sigma, learning_rate_op)

            # learning rates for all neurons, based on iteration number and location w.r.t. BMU.
            distanceSquare = tf.reduce_sum(tf.pow(tf.subtract(
                self.locations, tf.stack( [winnerLocation for _ in range(m * n)] ) ) , 2 ), 1)
            #gausian function for neghborhood
            neighbourhood = tf.exp(tf.negative(tf.div(tf.cast(
                distanceSquare, "float32"), tf.pow(sigma_op, 2))))
            #update learning rate
            learning_rate_op = tf.multiply(alpha_op, neighbourhood)

            # Update multiplyer for all the weights vectors of all neurons
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim]) for i in range(m * n)] )

            #update the  delta weight using learing rate
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self.vect_input for _ in range(m * n)]), self.weights))

            # W(t+1) = W(t) + W_delta
            #updating the weights
            new_weightages_op = tf.add(self.weights, weightage_delta)

            # Update weightge_vects by assigning new_weightages_op to it.
            self.training_op = tf.assign(self.weights, new_weightages_op)

            self.sess = tf.Session()
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

    #set up default locations
    def createLocations(self, m, n):
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    # trains the SOM with the input run through the weight traing for each input during each itteration
    def train(self, input_vects):
        for iter_no in range(self.n_iterations):
            for input_vect in input_vects:
                self.sess.run(self.training_op, 
                        feed_dict={self.vect_input: input_vect, self.iter_input: iter_no})

        #update weights and locations
        self.weightages = list(self.sess.run(self.weights))
        self.locations = list(self.sess.run(self.locations))

    #give locations to inputs in order to graph
    def map_vects(self, input_vects):
        to_return = []
        for vect in input_vects:
            min_index = min( [i for i in range(len(self.weightages))], 
                            key=lambda x: np.linalg.norm(vect - self.weightages[x]) )
            to_return.append(self.locations[min_index])

        return to_return

#end of SOM

#load mnist and create a subset of only 5s and 1s
mnist = fetch_mldata('MNIST original')

subSetX=[]
subSetY=[]
subSetXTest=[]
subSetYTest=[]
X=mnist.data
Y=mnist.target
# set up a subset with only values for 1 and 5 of size 60 ( 30 of each value) for training

index=7000 # start of images of ones
for j in range(index,index+61):
    subSetX.append(X[j])
    subSetY.append(Y[j])
index=35000 #start of images of 5s 
for j in range(index,index+61):
    subSetX.append(X[j])
    subSetY.append(Y[j])
# set up a subset with only values for 1 and 5 of size 10 ( 5 of each value) for testing

index=7100
for j in range(index,index+5):
    subSetXTest.append(X[j])
    subSetYTest.append(Y[j])
index=(7000*5)+100
for j in range(index,index+5):
    subSetXTest.append(X[j])
    subSetYTest.append(Y[j])

x_train =subSetX
y_train = subSetY
x_test =subSetXTest
y_test = subSetYTest

#set up and train the network **************
som = SOM(30, 30, 784, 100)
som.train(x_train)

#set up training data to be shown in a Plot
mapped = som.map_vects(x_train)
mappedarr = np.array(mapped)
x1 = mappedarr[:,0] 
y1 = mappedarr[:,1]

#test the network ***********************
mappedtest = som.map_vects(x_test)
mappedtestarr = np.array(mappedtest)
x2 = mappedtestarr[:,0]
y2 = mappedtestarr[:,1]



# calculate k-means ******************************
clusters=6
print("with clusters n=")
print(clusters)
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(subSetX)

centers = kmeans.cluster_centers_

# diplay the results in two Plots 
plt.figure(1, figsize=(12,6))
plt.subplot(121)

# Plot 1 training displayed on it's own to show areas
plt.scatter(x1,y1)
for i, m in enumerate(mapped):
    plt.text( m[0], m[1],y_train[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.title('Train MNIST 60')

#new plot 
plt.subplot(122)
# Plot 2: Training and tests overlap with clusters
#training
plt.scatter(x1,y1)
for i, m in enumerate(mapped):
    plt.text( m[0], m[1],y_train[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
#tests
plt.scatter(x2,y2)
for i, m in enumerate(mappedtest):
    plt.text( m[0], m[1],y_test[i], ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5, lw=0))
plt.title('Test MNIST 10 + Train MNIST 60')

#clusters
# map the cluster locations to the SOM
clusterMap=som.map_vects(centers)
clusterMapArr=np.array(clusterMap)
x3 = clusterMapArr[:,0]
y3 = clusterMapArr[:,1]
plt.scatter(x3,y3)
for i, m in enumerate(clusterMap):
     plt.text( m[0], m[1],"x", ha='center', va='center', bbox=dict(facecolor='blue', alpha=0.5, lw=0))
plt.show()




