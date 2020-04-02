def gradient(X, y, w, b):
    # Input:
    # X: nxd matrix
    # y: n-dimensional vector with labels (+1 or -1)
    # w: d-dimensional vector
    # b: a scalar bias term
    # Output:
    # wgrad: d-dimensional vector with gradient
    # bgrad: a scalar with gradient
    
    n, d = X.shape
    wgrad = np.zeros(d)
    bgrad = 0.0
    
    # YOUR CODE HERE
    div = (-y*sigmoid(-y*(X@w + b)))
    wgrad = np.dot(X.T, div)
    bgrad = np.sum(div)
             
    return wgrad, bgrad



def test_grad1():
    X = np.random.rand(25,5) # generate n random vectors with d dimensions
    w = np.random.rand(5) # define a random weight vector
    b = np.random.rand(1) # define a bias
    y = (np.random.rand(25)>0.5)*2-1 # set labels all-(+1)
    wgrad, bgrad = gradient(X, y, w, b) # compute the gradient using your function
    
    return wgrad.shape == w.shape and np.isscalar(bgrad)


def test_grad2():
    X = np.random.rand(25,5) # generate n random vectors with d dimensions
    w = np.random.rand(5) # define a random weight vector
    b = np.random.rand(1) # define a bias
    y = (np.random.rand(25)>0.5)*2-1 # set labels all-(+1)
    wgrad, bgrad = gradient(X, y, w, b) # compute the gradient using your function
    wgrad2, bgrad2 = gradient_grader(X, y, w, b) # compute the gradient using ground truth
    return np.linalg.norm(wgrad - wgrad2)<1e-06 and np.linalg.norm(bgrad - bgrad2) < 1e-06 # test if they match

def test_grad3():
    X = np.random.rand(25,5) # generate n random vectors with d dimensions
    y = (np.random.rand(25)>0.5)*2-1 # set labels all-(+1)
    w = np.random.rand(5) # define a random weight vector
    b = np.random.rand(1) 

    w_s = np.random.rand(5)*1e-05 # define tiny random step 
    b_s = np.random.rand(1)*1e-05 # define tiny random step 
    ll1 = log_loss(X,y,w+w_s, b+b_s)  # compute log-likelihood after taking a step
    
    ll = log_loss(X,y,w,b) # use Taylor's expansion to approximate new loss with gradient
    wgrad, bgrad =gradient(X,y,w,b) # compute gradient
    ll2=ll+ wgrad @ w_s + bgrad * b_s # take linear step with Taylor's approximation
    return np.linalg.norm(ll1-ll2)<1e-05 # test if they match

def test_grad4():
    w1, b1, losses1 = logistic_regression_grader(features, labels, 1000, 1e-03, gradient)
    w2, b2, losses2 = logistic_regression_grader(features, labels, 1000, 1e-03)
    return(np.abs(losses1[-1]-losses2[-1])<0.1)

runtest(test_grad1, 'test_grad1')
runtest(test_grad2, 'test_grad2')
runtest(test_grad3, 'test_grad3')
runtest(test_grad4, 'test_grad4')
