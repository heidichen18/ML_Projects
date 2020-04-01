def log_loss(X, y, w, b=0):
    # Input:
    # X: nxd matrix
    # y: n-dimensional vector with labels (+1 or -1)
    # w: d-dimensional vector
    # Output:
    # a scalar
    assert np.sum(np.abs(y))==len(y) # check if all labels in y are either +1 or -1
    
    # YOUR CODE HERE
    #nll = (-1)*np.sum(np.log(y_pred))
    nll =  np.sum(np.log(1+np.exp(-y*(X@w + b))))
      
    return nll



def test_logloss1():
    X = np.random.rand(25,5) # generate n random vectors with d dimensions
    w = np.random.rand(5) # define a random weight vector
    b = np.random.rand(1) # define a bias
    y = np.ones(25) # set labels all-(+1)
    ll=log_loss(X,y,w, b) # compute the probabilities of P(y=1|x;w) using your y_pred function
    # if labels are all-ones function becomes simply the sum of log of y_pred
    return np.isscalar(ll) # check whether the output is a scalar

def test_logloss2():
    X = np.random.rand(25,5) # generate n random vectors with d dimensions
    w = np.random.rand(5) # define a random weight vector
    b = np.random.rand(1) # define a bias
    y = np.ones(25) # set labels all-(+1)
    ll=log_loss(X,y,w, b) # compute the probabilities of P(y=1|x;w) using your y_pred function
    ll2=-np.sum(np.log(y_pred(X, w, b))) # if labels are all-ones function becomes simply the sum of log of y_pred
    return np.linalg.norm(ll-ll2)<1e-05

def test_logloss3():
    X = np.random.rand(25,5) # generate n random vectors with d dimensions
    w = np.random.rand(5) # define a random weight vector
    b = np.random.rand(1) # define a bias
    y = -np.ones(25) # set labels all-(-1)
    ll=log_loss(X,y,w,b) # compute the probabilities of P(y=1|x;w) using your y_pred function
    ll2=-np.sum(np.log(1-y_pred(X,w, b))) # if labels are all-ones function becomes simply the sum of log of 1-y_pred
    return np.linalg.norm(ll-ll2)<1e-05

def test_logloss4():
    X = np.random.rand(20,5) # generate n random vectors with d dimensions
    w = np.array([0,0,0,0,0]) # define an all-zeros weight vector
    y = (np.random.rand(20)>0.5)*2-1; # define n random labels (+1 or -1)
    ll=log_loss(X,y,w,0) # compute the probabilities of P(y=1|x;w) using your y_pred function
    # the log-likelihood for each of the 20 examples should be exactly 0.5:
    return np.linalg.norm(ll+20*np.log(0.5))<1e-08 

def test_logloss5():
    X = np.random.rand(500,15) # generate n random vectors with d dimensions
    w = np.random.rand(15) # define a random weight vector
    b = np.random.rand(1) # define a bias
    y = (np.random.rand(500)>0.5)*2-1; # define n random labels (+1 or -1)
    ll=log_loss(X,y,w,b) # compute the probabilities of P(y=1|x;w) using your y_pred function
    ll2=log_loss_grader(X,y,w,b) # compute the probabilities of P(y=1|x;w) using your y_pred function
    return np.linalg.norm(ll-ll2)<1e-05

runtest(test_logloss1, 'test_logloss1')
runtest(test_logloss2, 'test_logloss2')
runtest(test_logloss3, 'test_logloss3')
runtest(test_logloss4, 'test_logloss4')
runtest(test_logloss5, 'test_logloss5')
