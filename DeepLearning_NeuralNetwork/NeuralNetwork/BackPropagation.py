def backprop(W, A, Z, y):
    """
    
    INPUT:
    W weights (cell array)
    A output of forward pass (cell array)
    Z output of forward pass (cell array)
    y vector of size n (each entry is a label)
    
    OUTPUTS:
    
    gradient = the gradient with respect to W as a cell array of matrices
    """
    
    # Convert delta to a row vector to make things easier
    delta = (MSE_grad(Z[-1].flatten(), y) * 1).reshape(-1, 1)

    # compute gradient with backprop
    gradients = []
    
    # YOUR CODE HERE
    for i in range(len(W)-1, -1, -1):
        gradients.append((Z[i].T) @ (delta))
        delta = ReLU_grad(A[i]) * (delta@(W[i].T))
        
    gradients = gradients[::-1]
    
    return gradients



def backprop_test1():
    X, y = generate_data() # generate data
    
    n, _ = X.shape
    W = initweights([2, 3, 1]) # generate random weights
    A, Z = forward_pass(W, X)
    
    gradient = backprop(W, A, Z, y) # backprop to calculate the gradient
    
    # You should return a list with the same len as W
    return len(gradient) == len(W)

def backprop_test2():
    X, y = generate_data() # generate data
    
    n, _ = X.shape
    W = initweights([2, 3, 1]) # generate random weights
    A, Z = forward_pass(W, X)
    
    gradient = backprop(W, A, Z, y) # backprop to calculate the gradient
    
    # gradient[i] should match the shape of W[i]
    return np.all([gradient[i].shape == W[i].shape for i in range(len(W))])

def backprop_test3():
    X, y = generate_data() # generate data
    
    n, _ = X.shape
    
    # Use a one layer network
    # This is essentially the least squares
    W = initweights([2, 1]) 
    
    A, Z = forward_pass(W, X)
    
    # backprop to calculate the gradient
    gradient = backprop(W, A, Z, y) 
    
    # calculate the least square gradient
    least_square_gradient = 2 *((X.T @ X) @ W[0] - X.T @ y.reshape(-1, 1)) / n
    
    # gradient[0] should be the least square gradient
    return np.linalg.norm(gradient[0] - least_square_gradient) < 1e-7

def backprop_test4():
    X, y = generate_data() # generate data
    
    n, _ = X.shape
    
    # Use a one layer network
    # This is essentially the least squares
    W = initweights([2, 5, 5, 1]) 
    
    A, Z = forward_pass(W, X)
    
    # backprop to calculate the gradient
    gradient = backprop(W, A, Z, y) 
    
    # calculate the backprop gradient
    gradient_grader = backprop_grader(W, A, Z, y)
    
    # Check whether your gradient matches ours
    OK=[len(gradient_grader)==len(gradient)] # check if length matches
    for (g,gg) in zip(gradient_grader,gradient): # check if each component matches in shape and values
        OK.append(gg.shape==g.shape and (np.linalg.norm(g - gg) < 1e-7))
    return(all(OK))

def backprop_test5():
    # Here we reverse your gradient output and check that reverse with ours. It shouldn't match. 
    # If your reverse gradient matches our gradient, this means you outputted the gradient in reverse order.
    # This is a common mistake, as the loop is backwards. 
    X, y = generate_data() # generate data
    
    n, _ = X.shape
    
    # Use a one layer network
    # This is essentially the least squares
    W = initweights([2, 5, 5, 1]) 
    
    A, Z = forward_pass(W, X)
    
    # backprop to calculate the gradient
    gradient = backprop(W, A, Z, y) 
    
    # calculate the backprop gradient
    gradient_grader = backprop_grader(W, A, Z, y)

    gradient.reverse() # reverse the gradient. From now on it should NOT match
    # Check whether your gradient matches ours
    OK=[] # check if length matches
    for (g,gg) in zip(gradient_grader,gradient): # check if each component matches
        OK.append(gg.shape==g.shape and (np.linalg.norm(g - gg) < 1e-7))
    return(not all(OK)) 



runtest(backprop_test1, 'backprop_test1')
runtest(backprop_test2, 'backprop_test2')
runtest(backprop_test3, 'backprop_test3')
runtest(backprop_test4, 'backprop_test4')
runtest(backprop_test5, 'backprop_test5')
