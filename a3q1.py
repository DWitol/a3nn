import numpy as np
from scipy import stats
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
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
        if subSetX[i][j] == 0: arr.append(0)
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

    def activationFunc():
        for i in self.weights:
            inputs += self.weights[i].node.value * self.weights[i].weight
        if(inputs< 0): self.value = -1
        elif(inputs > 0): self.value = 1
        

class Network(object):
    def __init__(self,numNodes):
        self.nodes = []
        genNodes(numNodes)

    def genNodes(self,numNodes):
        while i in range(0,numNodes):
            self.nodes.append(Node())
        #;)
    def stimulateNetwork():

        for i in range(0,self.nodes.size()):
            self.nodes
