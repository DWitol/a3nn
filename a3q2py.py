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

print(tf.__version__)
mnist = fetch_mldata('MNIST original')
a = mnist.data.shape
b = mnist.target.shape
c = np.unique(mnist.target)
d = mnist.DESCR
print(a)
X=mnist.data
Y=mnist.target
# print(X[0])
print(70000/10)
print(Y[7000])
print(Y[14000])
print(Y[21000])
print(Y[28000])
print(Y[35000])

subSetX=[]
subSetY=[]

# set up a subset with only values for 1 to 5 of size 100 ( 20 of each value)
for i in range(1,6):
	index=7000*i
	for j in range(index,index+10):
		subSetX.append(X[j])
		subSetY.append(Y[j])

print(np.unique(subSetY))
print(len(subSetY))
# print(subSetX[0])


#replace all values in mnist with binary values
for i in range(0,len(subSetX)):
    arr = []
    for j in range(0,len(subSetX[i])):
        if subSetX[i][j] == 0: arr.append(0)
        else: arr.append(1)
    subSetX[i] = arr


clusters=5
print("with clusters n=")
print(clusters)
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(subSetX)

centers = kmeans.cluster_centers_

print(centers)
print(len(centers))
# I don't think we need this part for the question but im not sure 
# neigh=NearestNeighbors(n_neighbors=20)
# neigh.fit(subSetX)

# dists,indexs = neigh.kneighbors(X=centers)
# def getR(dists,k):
#     sumSquared=0
#     for i in dist:
#         sumSquared+=i*i
#     return math.sqrt(sumSquared/k)

# r=[]
# for dist in dists:
#     r.append(getR(dist,20))

# print(r)

class SOM(object):

    # To check if the SOM has been trained
    trained = False

    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):

        # Assign required variables first
        self.m = m; self.n = n
        if alpha is None:
            alpha = 0.2
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self.n_iterations = abs(int(n_iterations))

        self.graph = tf.Graph()
        with self.graph.as_default():

            # To save data, create weight vectors and their location vectors

            self.weightage_vects = tf.Variable(tf.random_normal( [m * n, dim]) )

            self.location_vects = tf.constant(np.array(list(self.neuron_locations(m, n))))

            # Training inputs

            # The training vector
            self.vect_input = tf.placeholder("float", [dim])
            # Iteration number
            self.iter_input = tf.placeholder("float")

            # Training Operation  # tf.stack result will be [ (m*n),  dim ]

            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self.weightage_vects, tf.stack(
                    [self.vect_input for _ in range(m * n)])), 2), 1)), 0) 
                    

            slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self.location_vects, slice_input, 
                        tf.constant(np.array([1, 2]), dtype=tf.int64) ), [2])

            # To compute the alpha and sigma values based on iteration number
            learning_rate_op = tf.subtract(1.0, tf.div(self.iter_input, self.n_iterations))
            alpha_op = tf.multiply(alpha, learning_rate_op)
            sigma_op = tf.multiply(sigma, learning_rate_op)

            # learning rates for all neurons, based on iteration number and location w.r.t. BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self.location_vects, tf.stack( [bmu_loc for _ in range(m * n)] ) ) , 2 ), 1)

            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(sigma_op, 2))))
            learning_rate_op = tf.multiply(alpha_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update the weightage vectors of all neurons
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim]) for i in range(m * n)] )

            ### Strucutre of updating weight ###
            ### W(t+1) = W(t) + W_delta ###
            ### wherer, W_delta = L(t) * ( V(t)-W(t) ) ###

            # W_delta = L(t) * ( V(t)-W(t) )
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self.vect_input for _ in range(m * n)]), self.weightage_vects))

            # W(t+1) = W(t) + W_delta
            new_weightages_op = tf.add(self.weightage_vects, weightage_delta)

            # Update weightge_vects by assigning new_weightages_op to it.
            self.training_op = tf.assign(self.weightage_vects, new_weightages_op)

            self.sess = tf.Session()
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

    def neuron_locations(self, m, n):

        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):

        # Training iterations
        for iter_no in range(self.n_iterations):
            # Train with each vector one by one
            for input_vect in input_vects:
                self.sess.run(self.training_op, 
                        feed_dict={self.vect_input: input_vect, self.iter_input: iter_no})

        # Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self.m)]
        self.weightages = list(self.sess.run(self.weightage_vects))
        self.locations = list(self.sess.run(self.location_vects))
        for i, loc in enumerate(self.locations):
            centroid_grid[loc[0]].append(self.weightages[i])

        self.centroid_grid = centroid_grid

        self.trained = True

    def get_centroids(self):

        if not self.trained:
            raise ValueError("SOM not trained yet")
        return self.centroid_grid

    def map_vects(self, input_vects):

        if not self.trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in input_vects:
            min_index = min( [i for i in range(len(self.weightages))], 
                            key=lambda x: np.linalg.norm(vect - self.weightages[x]) )
            to_return.append(self.locations[min_index])

        return to_return

import matplotlib.pyplot as plt
import numpy as np
import random as ran

## Applying SOM into Mnist data

mnist = fetch_mldata('MNIST original')

def train_size(num):
    x_train = mnist.data[:num]
    y_train = mnist.target[:num]
    return x_train, y_train

subSetX=[]
subSetY=[]
subSetXTest=[]
subSetYTest=[]
X=mnist.data
Y=mnist.target
# set up a subset with only values for 1 to 5 of size 100 ( 10 of each value) for training
for i in range(1,6):
    index=7000*i
    for j in range(index,index+10):
        subSetX.append(X[j])
        subSetY.append(Y[j])
# set up a subset with only values for 1 to 5 of size 100 ( 10 of each value) for testing
for i in range(1,6):
    index=7100*i
    for j in range(index,index+10):
        subSetXTest.append(X[j])
        subSetYTest.append(Y[j])

x_train =subSetX
y_train = subSetY
x_test =subSetXTest
y_test = subSetYTest

som = SOM(30, 30, 784, 20)
som.train(x_train)

# Fit train data into SOM lattice
mapped = som.map_vects(x_train)
mappedarr = np.array(mapped)
x1 = mappedarr[:,0]; y1 = mappedarr[:,1]


## Plots: 1) Train 2) Test+Train ###

plt.figure(1, figsize=(12,6))
plt.subplot(121)
# Plot 1 for Training only
plt.scatter(x1,y1)
# Just adding text
for i, m in enumerate(mapped):
    plt.text( m[0], m[1],y_train[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.title('Train MNIST 100')

# Testing
mappedtest = som.map_vects(x_test)
mappedtestarr = np.array(mappedtest)
x2 = mappedtestarr[:,0]
y2 = mappedtestarr[:,1]

plt.subplot(122)
# Plot 2: Training + Testing
plt.scatter(x1,y1)
# Just adding text
for i, m in enumerate(mapped):
    plt.text( m[0], m[1],y_train[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))

plt.scatter(x2,y2)
# Just adding text
for i, m in enumerate(mappedtest):
    plt.text( m[0], m[1],y_test[i], ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5, lw=0))
plt.title('Test MNIST 10 + Train MNIST 100')

plt.show()









