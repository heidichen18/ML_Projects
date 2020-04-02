import numpy as np
import os
from pylab import *
import matplotlib.pyplot as plt
%matplotlib inline 

from helper import *

print('You\'re running python %s' % sys.version.split(' ')[0])


np.random.seed(12)
n_samples = 500

class_one = np.random.multivariate_normal([5, 10], [[1, .25],[.25, 1]], n_samples)
class_one_labels = -np.ones(n_samples)

class_two = np.random.multivariate_normal([0, 5], [[1, .25],[.25, 1]], n_samples)
class_two_labels = np.ones(n_samples)

features = np.vstack((class_one, class_two))
labels = np.hstack((class_one_labels, class_two_labels)i)



# Let's see what what our feature arrays look like. 
features



# We can visualize these data distributions
plt.figure(figsize=(9, 6))
plt.scatter(features[:, 0], features[:, 1],
            c = labels, alpha = .6);

plt.title("Binary labeled data in 2D", size=15);
plt.xlabel("Feature 1", size=13);
plt.ylabel("Feature 2", size=13);


