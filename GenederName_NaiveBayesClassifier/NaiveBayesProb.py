def naivebayesPY(X, Y):
    """
    naivebayesPY(Y) returns [pos,neg]

    Computation of P(Y)
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (nx1)

    Output:
        pos: probability p(y=1)
        neg: probability p(y=-1)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    Y = np.concatenate([Y, [-1,1]])
    n = len(Y)
    # YOUR CODE HERE
    pos = np.count_nonzero(Y == 1) / n
    neg = np.count_nonzero(Y == -1) / n
    
    return pos,neg


pos, neg = naivebayesPY(X,Y)




# The following tests will check that the probabilities returned by your function sum to 1 (test1) and return the correct probabilities for a given set of input vectors (tests 2-4)

# Check that probabilities sum to 1
def naivebayesPY_test1():
    pos, neg = naivebayesPY(X,Y)
    return np.linalg.norm(pos + neg - 1) < 1e-5

# Test the Naive Bayes PY function on a simple example
def naivebayesPY_test2():
    x = np.array([[0,1],[1,0]])
    y = np.array([-1,1])
    pos, neg = naivebayesPY(x,y)
    pos0, neg0 = .5, .5
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    return test < 1e-5

# Test the Naive Bayes PY function on another example
def naivebayesPY_test3():
        x = np.array([[0,1,1,0,1],
            [1,0,0,1,0],
            [1,1,1,1,0],
            [0,1,1,0,1],
            [1,0,1,0,0],
            [0,0,1,0,0],
            [1,1,1,0,1]])    
        y = np.array([1,-1, 1, 1,-1,-1, 1])
        pos, neg = naivebayesPY(x,y)
        pos0, neg0 = 5/9., 4/9.
        test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
        return test < 1e-5

# Tests plus-one smoothing
def naivebayesPY_test4():
    x = np.array([[0,1,1,0,1],[1,0,0,1,0]])    
    y = np.array([1,1])
    pos, neg = naivebayesPY(x,y)
    pos0, neg0 = 3/4., 1/4.
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    return test < 1e-5    
        
    
runtest(naivebayesPY_test1, 'naivebayesPY_test1')
runtest(naivebayesPY_test2,'naivebayesPY_test2')
runtest(naivebayesPY_test3,'naivebayesPY_test3')
runtest(naivebayesPY_test4,'naivebayesPY_test4')
