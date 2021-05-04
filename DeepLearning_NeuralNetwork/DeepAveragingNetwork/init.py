%load_ext autoreload


%autoreload 2
# import PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data

# Set the seed for PyTorch random number generator
torch.manual_seed(1)

# If gpu is supported, then seed the gpu random number generator as well
gpu_available = torch.cuda.is_available()
if gpu_available:
    torch.cuda.manual_seed(1)
    


import sys
import matplotlib.pyplot as plt
from helper import *

%matplotlib notebook
print('You\'re running python %s' % sys.version.split(' ')[0])
print ("GPU is available:",gpu_available)




