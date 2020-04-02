import numpy as np
import matplotlib 
import sys
import matplotlib.pyplot as plt
import time
sys.path.append('/home/codio/workspace/.guides/hf')
from helper import *


%matplotlib notebook
print('You\'re running python %s' % sys.version.split(' ')[0])


def perceptron_update(x,y,w):
    """
    function w=perceptron_update(x,y,w);
    
    Implementation of Perceptron weights updating
    Input:
    x : input vector of d dimensions (d)
    y : corresponding label (-1 or +1)
    w : weight vector of d dimensions
    
    Output:
    w : weight vector after updating (d)
    """
    
    # YOUR CODE HERE
    w += np.dot(np.transpose(x), y)
    return w
    
# little test
x = np.random.rand(10)
y = -1
w = np.random.rand(10)
w1 = perceptron_update(x,y,w)



# This self test will check that your perceptron_update function returns the correct values for input vector [0,1], label -1, and weight vector [1,1]

def test_perceptron_update1():
    x = np.array([0,1])
    y = -1
    w = np.array([1,1])
    w1 = perceptron_update(x,y,w)
    return (w1.reshape(-1,) == np.array([1,0])).all()

def test_perceptron_update2(): 
    x = np.random.rand(25)
    y = 1
    w = np.zeros(25)
    w1 = perceptron_update(x,y,w)
    return np.linalg.norm(w1-x)<1e-8


def test_perceptron_update3():
    x = np.random.rand(25)
    y = -1
    w = np.zeros(25)
    w1 = perceptron_update(x,y,w)
    return np.linalg.norm(w1+x)<1e-8


runtest(test_perceptron_update1, 'test_perceptron_update1')
runtest(test_perceptron_update2, 'test_perceptron_update2')
runtest(test_perceptron_update3, 'test_perceptron_update3')



