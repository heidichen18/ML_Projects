import numpy as np
import sys
sys.path.append('/home/codio/workspace/.guides/hf')
from helper import *

%matplotlib inline
print('You\'re running python %s' % sys.version.split(' ')[0])


def hashfeatures(baby, B, FIX):
    """
    Input:
        baby : a string representing the baby's name to be hashed
        B: the number of dimensions to be in the feature vector
        FIX: the number of chunks to extract and hash from each string
    
    Output:
        v: a feature vector representing the input string
    """
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "suffix" + baby[-1*m:]
        v[hash(featurestring) % B] = 1
    return v



def name2features(filename, B=128, FIX=3, LoadFile=True):
    """
    Output:
        X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], B, FIX)
    return X



def genTrainFeatures(dimension=128):
    """
    Input: 
        dimension: desired dimension of the features
    Output: 
        X: n feature vectors of dimensionality d (nxd)
        Y: n labels (-1 = girl, +1 = boy) (n)
    """
    
    # Load in the data
    Xgirls = name2features("girls.train", B=dimension)
    Xboys = name2features("boys.train", B=dimension)
    X = np.concatenate([Xgirls, Xboys])
    
    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])
    
    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])
    
    return X[ii, :], Y[ii]




X, Y = genTrainFeatures(128)
