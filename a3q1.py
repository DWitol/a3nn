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
#replace all values in mnist with binary values
print(mnist.data[1])
for i in range(0,mnist.data.size):
    arr = []
    for j in range(0,mnist.data[i].size):
        if mnist.data[i][j] == 0: arr.append(0)
        else: arr.append(1)
    mnist.data[i] = arr
