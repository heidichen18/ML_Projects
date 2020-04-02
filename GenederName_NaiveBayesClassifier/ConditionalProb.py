def naivebayesPXY(X,Y):
    """
    naivebayesPXY(X, Y) returns [posprob,negprob]
    
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (n)
    
    Output:
        posprob: probability vector of p(x_alpha = 1|y=1)  (d)
        negprob: probability vector of p(x_alpha = 1|y=-1) (d)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = X.shape
    X = np.concatenate([X, np.ones((2,d)), np.zeros((2,d))])
    Y = np.concatenate([Y, [-1,1,-1,1]])
    n, d = X.shape
    
    # YOUR CODE HERE
    posprob = np.mean(X[Y == 1], axis=0)
    negprob = np.mean(X[Y == -1], axis=0)
    
    return posprob, negprob
    

posprob, negprob = naivebayesPXY(X,Y)



# The following tests check that your implementation of naivebayesPXY returns the same posterior probabilities as the correct implementation, in the correct dimensions

# test a simple toy example with two points (one positive, one negative)
def naivebayesPXY_test1():
    x = np.array([[0,1],[1,0]])
    y = np.array([-1,1])
    pos, neg = naivebayesPXY(x,y)
    pos0, neg0 = naivebayesPXY_grader(x,y)
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    return test < 1e-5

# test the probabilities P(X|Y=+1)
def naivebayesPXY_test2():
    pos, neg = naivebayesPXY(X,Y)
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    test = np.linalg.norm(pos - posprobXY) 
    return test < 1e-5

# test the probabilities P(X|Y=-1)
def naivebayesPXY_test3():
    pos, neg = naivebayesPXY(X,Y)
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    test = np.linalg.norm(neg - negprobXY)
    return test < 1e-5


# Check that the dimensions of the posterior probabilities are correct
def naivebayesPXY_test4():
    pos, neg = naivebayesPXY(X,Y)
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    return pos.shape == posprobXY.shape and neg.shape == negprobXY.shape

runtest(naivebayesPXY_test1,'naivebayesPXY_test1')
runtest(naivebayesPXY_test2,'naivebayesPXY_test2')
runtest(naivebayesPXY_test3,'naivebayesPXY_test3')
runtest(naivebayesPXY_test4,'naivebayesPXY_test4')
