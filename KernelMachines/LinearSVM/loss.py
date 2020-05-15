def loss(w, b, xTr, yTr, C):
    """
    INPUT:
    w     : d   dimensional weight vector
    b     : scalar (bias)
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    C     : scalar (constant that controls the tradeoff between l2-regularizer and hinge-loss)
    
    OUTPUTS:
    loss     : the total loss obtained with (w, b) on xTr and yTr (scalar)
    """
    
    loss_val = 0.0
    
    # YOUR CODE HERE
    margin = yTr*(xTr @ w + b)
    squaredHingeLoss = C * np.sum(np.maximum(1-margin, 0)**2)
    regularizer = w.T @ w
    
    loss_val = regularizer + squaredHingeLoss
    
    return loss_val


# These tests test whether your loss() is implemented correctly

xTr_test, yTr_test = generate_data()
n, d = xTr_test.shape

# Check whether your loss() returns a scalar
def loss_test1():
    w = np.random.rand(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 10)    
    return np.isscalar(loss_val)

# Check whether your loss() returns a nonnegative scalar
def loss_test2():
    w = np.random.rand(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 10)
    
    return loss_val >= 0

# Check whether you implement l2-regularizer correctly
def loss_test3():
    w = np.random.rand(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 0)
    loss_val_grader = loss_grader(w, b, xTr_test, yTr_test, 0)
    
    return (np.linalg.norm(loss_val - loss_val_grader) < 1e-5)

# Check whether you implemented the squared hinge loss and not the standard hinge loss
# Note, loss_grader_wrong is the wrong implementation of the standard hinge-loss, 
# so the results should NOT match.
def loss_test4():
    w = np.random.randn(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 1)
    badloss = loss_grader_wrong(w, b, xTr_test, yTr_test, 1)    
    return not(np.linalg.norm(loss_val - badloss) < 1e-5)


# Check whether you implement square hinge loss correctly
def loss_test5():
    w = np.random.randn(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 10)
    loss_val_grader = loss_grader(w, b, xTr_test, yTr_test, 10)
    
    return (np.linalg.norm(loss_val - loss_val_grader) < 1e-5)

# Check whether you implement loss correctly
def loss_test6():
    w = np.random.randn(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 100)
    loss_val_grader = loss_grader(w, b, xTr_test, yTr_test, 100)
    
    return (np.linalg.norm(loss_val - loss_val_grader) < 1e-5)

runtest(loss_test1,'loss_test1')
runtest(loss_test2,'loss_test2')
runtest(loss_test3,'loss_test3')
runtest(loss_test4,'loss_test4')
runtest(loss_test5,'loss_test5')
runtest(loss_test6,'loss_test6')
