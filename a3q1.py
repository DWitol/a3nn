import numpy as np
from scipy import stats
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
from pylab import imshow, cm, show
import math
import random
import time

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
	for j in range(index,index+20):
		subSetX.append(X[j])
		subSetY.append(Y[j])

print(np.unique(subSetY))
print(len(subSetY))
# print(subSetX[0])


#replace all values in mnist with binary values
for i in range(0,len(subSetX)):
    arr = []
    for j in range(0,len(subSetX[i])):
        if subSetX[i][j] == 0: arr.append(-1)
        else: arr.append(1)
    subSetX[i] = arr


# print(subSetX[0])
class pair(object):
    def __init__(self, node, weight):
        self.node = node
        self.weight = weight

class Node(object):
    def __init__(self):
        self.weights = []
        self.value = 0

    def activationFunc(self):
        inputs = 0
        for i in range(0,len(self.weights)):
            inputs += self.weights[i][0].value * self.weights[i][1]
        if(inputs< 0): self.value = -1
        elif(inputs > 0): self.value = 1

        return self.value

    def connectNode(self, OtherNode,weight):
        self.weights.append([OtherNode,weight])




class Network(object):
    def genNodes(self,numNodes):
        while i in range(0,numNodes):
            self.nodes.append(Node())

    def __init__(self,numNodes):
        self.nodes = []
        for i in range(0,numNodes):
            self.nodes.append(Node())

    def setNodes(self,pattern):
        for i in range(0,len(pattern)):
            self.nodes[i].value = pattern[i]

        # for i in range(0,numNodes):
        #     for i in self.nodes:
        #         for j in self.nodes:


        #;)
    def stimulateNetwork(self):
        changed = 1
        circulation =0
        while(changed == 1):
            print(circulation)
            circulation += 1
            changed = 0
            for i in range(0,len(self.nodes)):
                oldValue = self.nodes[i].value
                newValue = self.nodes[i].activationFunc()
                if(oldValue != newValue):changed = 1

        print("network stimulated")

def recall(W, patterns, steps=5):
    sgn = np.vectorize(lambda x: -1 if x<0 else +1)
    for _ in range(steps):
        patterns = sgn(np.dot(patterns,W))
    return patterns

hopfieldNetwork = Network(784)
r,c = np.matrix(subSetX).shape
W = np.zeros((c,c))
for i in np.matrix(subSetX):
    W = W + np.outer(i,i)

W[np.diag_indices(c)] = 0
W = W/r
bob = recall(W,subSetX)
print(W[0].size)
for i in range(0,W[0].size):
    for j in range(0,W[0].size):
        hopfieldNetwork.nodes[i].connectNode(hopfieldNetwork.nodes[j], W[i][j])

hopfieldNetwork.setNodes(subSetX[0])
hopfieldNetwork.stimulateNetwork()
