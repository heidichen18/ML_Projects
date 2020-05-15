import numpy as np
from numpy.matlib import repmat
import sys
import time

from helper import *

import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import linregress

import pylab
from matplotlib.animation import FuncAnimation

%matplotlib notebook
print('You\'re running python %s' % sys.version.split(' ')[0])

xTr, yTr = generate_data()
visualize_2D(xTr, yTr)

