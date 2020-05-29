# import PyTorch and its subpackages
import torch
import torch.nn as nn
from torch.nn import functional as F

# Also other packages for convenience
import numpy as np
import helper as h
import matplotlib.pyplot as plt

X = torch.zeros(3, 2) # this syntax is similar to NumPy. In Numpy, we would do np.zero(3,2)

print(X)
print(X.shape)

X_numpy = np.arange(15)
print('NumPy Array: ', X_numpy)

X_tensor = torch.tensor(X_numpy)
print('Pytorch Tensor: ', X_tensor)

X_tensor_numpy = X_tensor.numpy()

print(X_tensor_numpy)
print(type(X_tensor_numpy))


# Create two numpy arrays
A = np.array([1, 2, 3])
B = np.array([1, 5, 6])

# Convert the two numpy arrays into torch Tensor
A_tensor = torch.Tensor(A)
B_tensor = torch.Tensor(B)

# addition / subtraction
print('Addition in Numpy', A + B)
print('Addition in PyTorch', A_tensor + B_tensor)
print()

# scalar multiplication
print('Scalar Multiplication in Numpy', 3*A)
print('Scalar Multiplication in PyTorch', 3*A_tensor)
print()

# elementwise multiplication
print('Elementwise Multiplication in Numpy', A*B)
print('Elementiwse Multiplication in PyTorch', A_tensor*B_tensor)
print()

# matrix multiplication
# this is slightly different from NumPy
print('Elementwise Multiplication in Numpy', A@B)
print('Elementiwse Multiplication in PyTorch', torch.matmul(A_tensor, B_tensor))
print()

# Elementwise comparison
print('NumPy: ', A == B)
print('PyTorch: ', A_tensor == B_tensor)
print()

# Generate a random matrix
C = np.array([[10, 9, 8], [6, 7, 5], [1, 2, 3]])
C_tensor = torch.Tensor(C)

print('C', C)
print()

# Sum along the row
# In NumPy, we specify the axis.
# In PyTorch, we specify the dim
print('NumPy: ', np.sum(C, axis=0))
print('PyTorch: ' ,torch.sum(C_tensor, dim=0))
print()

# Find the mean along the column
# In NumPy, we specify the axis.
# In PyTorch, we specify the dim
print('NumPy: ', np.mean(C, axis=1))
print('PyTorch: ', torch.mean(C_tensor, dim=1))
print()


# Find the argmax along the column
# In NumPy, we specify the axis.
# In PyTorch, we specify the dim
print('NumPy: ', np.argmax(C, axis=1))
print('PyTorch: ', torch.argmax(C_tensor, dim=1))
print()



