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
# print(70000/10)
# print(Y[7000])
# print(Y[14000])
# print(Y[21000])
# print(Y[28000])
# print(Y[35000])

subSetX=[]
subSetY=[]
subSetXTest=[]
subSetYTest=[]
X=mnist.data
Y=mnist.target

# set up a subset with only values for 1 and 5 of size 60 ( 30 of each value) for training

index=7000 # start of images of ones
for j in range(index,index+2):
    subSetX.append(X[j])
    subSetY.append(Y[j])
index=35000 #start of images of 5s
for j in range(index,index+2):
    subSetX.append(X[j])
    subSetY.append(Y[j])
# set up a subset with only values for 1 and 5 of size 10 ( 5 of each value) for testing

index=7100
for j in range(index,index+3):
    subSetXTest.append(X[j])
    subSetYTest.append(Y[j])
index=(7000*5)+100
for j in range(index,index+3):
    subSetXTest.append(X[j])
    subSetYTest.append(Y[j])

x_train =subSetX
y_train = subSetY
x_test =subSetXTest
y_test = subSetYTest
print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

# print(subSetX[0])
#subSetX = x_test
for i in range(0,len(x_train)):
    arr = []
    for j in range(0,len(x_train[0])):
        if x_train[i][j] == 0: arr.append(-1)
        else: arr.append(1)
    x_train[i] = arr
    print(i," of ", len(x_train))

for i in range(0,len(x_test)):
    arr = []
    for j in range(0,len(x_test[0])):
        if x_test[i][j] == 0: arr.append(-1)
        else: arr.append(1)
    x_test[i] = arr
    print(i," of ", len(x_test))

# def to_pattern(letter):
#     from numpy import array
#     return array([+1 if c=='X' else -1 for c in letter.replace('\n','')])

def display(pattern):
    imshow(pattern.reshape((28,28)),cmap=cm.binary, interpolation='nearest')
    show()

#replace all values in mnist with binary values

# print(subSetX[0])
class Pair(object):
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
        pair = Pair(OtherNode,weight)
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

    def getState(self):
        arr = []
        for i in self.nodes:
            arr.append(i.value)

        return arr
        #;)
    def stimulateNetwork(self):
        changed = 1
        while(changed == 1):
            changed = 0
            for i in range(0,len(self.nodes)):
                oldValue = self.nodes[i].value
                newValue = self.nodes[i].activationFunc()
                if(oldValue != newValue):changed = 1


def recall(W, patterns, steps=20):
    sgn = np.vectorize(lambda x: -1 if x<0 else +1)
    for _ in range(steps):
        patterns = sgn(np.dot(patterns,W))
    return patterns

#for i in x_test:
#    display(np.matrix(i))

hopfieldNetwork = Network(784)
#training
print(y_train[0])
r,c = np.matrix(x_train).shape
W = np.zeros((c,c))
for i in np.matrix(x_train):
    display(i)
    W = W + np.outer(i,i)
W[np.diag_indices(c)] = 0
W = W/r

#bob = recall(W,x_train)
print(W[0].size)
for i in range(0,W[0].size):
    for j in range(0,W[0].size):
        hopfieldNetwork.nodes[i].connectNode(hopfieldNetwork.nodes[j], W[i][j])
createdPatterns = 1
states = []

hopfieldNetwork.setNodes(x_train[1])
hopfieldNetwork.stimulateNetwork()
state1 = hopfieldNetwork.getState()
display(np.matrix(X[35000]))
display(np.matrix(state1))

for i in range(0,len(x_test)):
    display(recall(W,x_test[i]));

# for i in range(0,len(X)-1):
#     hopfieldNetwork.setNodes(X[i])
#     hopfieldNetwork.stimulateNetwork()
#     state1 = hopfieldNetwork.getState()
#     hopfieldNetwork.setNodes(X[i+1])
#     hopfieldNetwork.stimulateNetwork()
#     state2 = hopfieldNetwork.getState()
#     print("made it through pics")
