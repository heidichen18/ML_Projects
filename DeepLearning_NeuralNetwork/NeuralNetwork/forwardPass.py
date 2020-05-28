def initweights(specs):
    """
    Given a specification of the neural network, output a random weight array
    INPUT:
        specs - array of length m+1. specs[0] should be the dimension of the feature and spec[-1] 
                should be the dimension of output
    
    OUTPUT:
        W - array of length m, each element is a matrix
            where size(weights[i]) = (specs[i], specs[i+1])
    """
    W = []
    for i in range(len(specs) - 1):
        W.append(np.random.randn(specs[i], specs[i+1]))
    return W


# If we want to create a network that 
#   i) takes in feature of dimension 2
#   ii) has 1 hidden layer with 3 hidden units
#   iii) output a scalar
# then we initialize the the weights the following way:

W = initweights([2, 3, 1])


def forward_pass(W, xTr):
    """
    function forward_pass(weights,xTr)
    
    INPUT:
    W - an array of L weight matrices
    xTr - nxd matrix. Each row is an input vector
    
    OUTPUTS:
    A - a list of matrices (of length L) that stores result of matrix multiplication at each layer 
    Z - a list of matrices (of length L) that stores result of transition function at each layer 
    """
    
    # Initialize A and Z
    A = [xTr]
    Z = [xTr]
    # YOUR CODE HERE
    for i in range(len(W)):
        a = Z[i] @ W[i]
        A.append(a)
        
        if i < (len(W)-1):
            z = ReLU(a)
        else:
            z = a
            
        Z.append(z)    
    
    
    return A, Z



def forward_test1():
    X, _ = generate_data() # generate data
    W = initweights([2, 3, 1]) # generate random weights
    out = forward_pass(W, X) # run forward pass
    return len(out) == 2 # make sure that your function return a tuple

def forward_test2():
    X, _ = generate_data() # generate data
    W = initweights([2, 3, 1]) # generate random weights
    A, Z = forward_pass(W, X) # run forward pass
    return len(A) == 3 and len(Z) == 3 # Make sure that output produced match the length of the weight

def forward_test3():
    X, _ = generate_data() # generate data
    n, _ = X.shape
    W = initweights([2, 3, 1]) # generate random weights
    A, Z = forward_pass(W, X) # run forward pass
    return (A[1].shape == (n, 3) and 
            Z[1].shape == (n, 3)  and
            A[2].shape == (n, 1) and
            A[2].shape == (n, 1) ) # Make sure the layer produce the right shape output

def forward_test4():
    X = -1*np.ones((1, 2)) # generate a feature matrix of all negative ones
    W = [np.ones((2, 1))] # a single layer network with weights one
    A, Z = forward_pass(W, X) # run forward pass
    
    # check whether you do not apply the transition function to A[-1] 
    return np.linalg.norm(Z[-1] - X@W[0]) < 1e-7

def forward_test5():
    X, _ = generate_data() # generate data
    n, _ = X.shape
    W = initweights([2, 3, 1]) # generate random weights
    A, Z = forward_pass(W, X) # run your forward pass
    A_grader, Z_grader = forward_pass_grader(W, X) # run our forward pass
    
    Adiff = 0
    Zdiff = 0
    
    # compute the difference between your solution and ours
    for i in range(1, 3):
        Adiff += np.linalg.norm(A[i] - A_grader[i])
        Zdiff += np.linalg.norm(Z[i] - Z_grader[i])
        
    return Adiff < 1e-7 and Zdiff < 1e-7

runtest(forward_test1, "forward_test1")
runtest(forward_test2, "forward_test2")
runtest(forward_test3, "forward_test3")
runtest(forward_test4, "forward_test4")
runtest(forward_test5, "forward_test5")
