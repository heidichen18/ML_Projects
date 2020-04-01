def y_pred(X, w, b=0):
    # Input:
    # X: nxd matrix
    # w: d-dimensional vector
    # b: scalar (optional, if not passed on is treated as 0)
    # Output:
    # prob: n-dimensional vector
    
    # YOUR CODE HERE
    prob = sigmoid((np.dot(w,X.T)+b))
    
    return prob




def test_ypred1():
    n = 20
    d = 5
    X = np.random.rand(n,d) # generate n random vectors with d dimensions
    w = np.random.rand(5) # define a random weight vector
    probs=y_pred(X,w,0) # compute the probabilities of P(y=1|x;w) using your y_pred function
    return probs.shape == (n, ) # check if all outputs are >=0 and <=1


def test_ypred2():
    n = 20
    d = 5
    X = np.random.rand(n,d) # generate n random vectors with d dimensions
    w = np.random.rand(5) # define a random weight vector
    probs=y_pred(X, w, 0) # compute the probabilities of P(y=1|x;w) using your y_pred function
    return all(probs>=0) and all(probs<=1) # check if all outputs are >=0 and <=1

def test_ypred3():
    n = 20
    d = 5
    X = np.random.rand(n,d) # generate n random vectors with d dimensions
    w = np.random.rand(5) # define a random weight vector
    probs1=y_pred(X, w, 0) # compute the probabilities of P(y=1|x;w) using your y_pred function
    probs2=y_pred(X,-w, 0) # compute the probabilities of P(y=1|x;w) using your y_pred function
    return np.linalg.norm(probs1+probs2-1)<1e-08 # check if P(y|x;w)+P(y|x;-w)=1



def test_ypred4():
    X=np.random.rand(25,4) # define random input
    w=np.array([1,0,0,0]) # all-zeros weight vector
    prob=y_pred(X, w, 0) # compute P(y|X;w)
    truth=sigmoid(X[:,0]) # should simply be the sigmoid of the first feature
    return np.linalg.norm(prob-truth)<1e-08 # see if they match


def test_ypred5(): 
    X=np.array([[0.61793598, 0.09367891], # define 3 inputs (2D)
               [0.79447745, 0.98605996],
               [0.53679997, 0.4253885 ]])
    w=np.array([0.9822789 , 0.16017851]); # define weight vector
    prob=y_pred(X, w, 3) # compute P(y|X;w)
    truth=np.array([0.97396645,0.98089179,0.97328431]) # this is the grount truth
    return np.linalg.norm(prob-truth)<1e-08 # see if they match

runtest(test_ypred1, 'test_ypred1')
runtest(test_ypred2, 'test_ypred2')
runtest(test_ypred3, 'test_ypred3')
runtest(test_ypred4, 'test_ypred4')
runtest(test_ypred5, 'test_ypred5')
