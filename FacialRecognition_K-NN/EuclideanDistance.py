import numpy as np
from scipy.stats import mode
import sys
%matplotlib notebook
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
from helper import *

print('You\'re running python %s' % sys.version.split(' ')[0])

def l2distance(X,Z=None):
    # function D=l2distance(X,Z)
    #
    # Computes the Euclidean distance matrix.
    # Syntax:
    # D=l2distance(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    #
    # call with only one input:
    # l2distance(X)=l2distance(X,X)
    #
    if Z is None:
        Z=X;

    n,d1=X.shape
    m,d2=Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"

    # YOUR CODE HERE
    G=np.dot(X,np.transpose(Z))
    X_ones = np.ones((1,d1))
    Z_ones = np.ones((1,d1))
    xi = np.transpose(np.dot(X_ones,np.transpose(np.square(X))))
    zj = np.dot(Z_ones, np.transpose(np.square(Z)))
    S=np.zeros((n,m))+xi
    R=np.zeros((n,m))+zj
    D2=abs(S-2*G+R)
    D=np.sqrt(D2)
   
    return D
