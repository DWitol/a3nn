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
#this som is baised on the som from https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
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

            # To save data, create weight vectors and their location vectors
            self.weightage_vects = tf.Variable(tf.random_normal( [m * n, dim]) )
            self.location_vects = tf.constant(np.array(list(self.neuron_locations(m, n))))
            # Training inputs
            self.vect_input = tf.placeholder("float", [dim])
            self.iter_input = tf.placeholder("float")

            #Gets the winner index 
            winnerIndex = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self.weightage_vects, tf.stack(
                    [self.vect_input for _ in range(m * n)])), 2), 1)), 0) 
                    
            #gets the winner from the index
            slice_input = tf.pad(tf.reshape(winnerIndex, [1]), np.array([[0, 1]]))
            winnerLocation = tf.reshape(tf.slice(self.location_vects, slice_input, 
                        tf.constant(np.array([1, 2]), dtype=tf.int64) ), [2])

            # compute new alpha and sigma for each iteration
            learning_rate_op = tf.subtract(1.0, tf.div(self.iter_input, self.n_iterations))
            alpha_op = tf.multiply(alpha, learning_rate_op)
            sigma_op = tf.multiply(sigma, learning_rate_op)

            # learning rates for all neurons, based on iteration number and location w.r.t. BMU.
            distanceSquare = tf.reduce_sum(tf.pow(tf.subtract(
                self.location_vects, tf.stack( [winnerLocation for _ in range(m * n)] ) ) , 2 ), 1)
            #gausian function for neghborhood
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                distanceSquare, "float32"), tf.pow(sigma_op, 2))))
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
        for iter_no in range(self.n_iterations):
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

    def get_centroids(self):
        return self.centroid_grid

    def map_vects(self, input_vects):
        to_return = []
        for vect in input_vects:
            min_index = min( [i for i in range(len(self.weightages))], 
                            key=lambda x: np.linalg.norm(vect - self.weightages[x]) )
            to_return.append(self.locations[min_index])

        return to_return


## Applying SOM into Mnist data

mnist = fetch_mldata('MNIST original')

subSetX=[]
subSetY=[]
subSetXTest=[]
subSetYTest=[]
X=mnist.data
Y=mnist.target
# set up a subset with only values for 1 and 5 of size 60 ( 30 of each value) for training

index=7000
for j in range(index,index+30):
    subSetX.append(X[j])
    subSetY.append(Y[j])
index=7000*5
for j in range(index,index+30):
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

som = SOM(30, 30, 784, 100)
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

clusters=5
print("with clusters n=")
print(clusters)
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(subSetX)

centers = kmeans.cluster_centers_

print(centers)
print(len(centers))

# reduced_data = subSetX
# kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10)
# kmeans.fit(reduced_data)

# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')

# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()







