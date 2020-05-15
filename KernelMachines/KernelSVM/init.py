import numpy as np
from helper import *
import matplotlib.pyplot as plt
import sys

print('You\'re running python %s' % sys.version.split(' ')[0])


%matplotlib notebook
xTr,yTr = generate_data()
visualize_2D(xTr, yTr)



