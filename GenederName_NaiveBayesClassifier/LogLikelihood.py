def loglikelihood(posprob, negprob, X_test, Y_test):
    """
    loglikelihood(posprob, negprob, X_test, Y_test) returns loglikelihood of each point in X_test
    
    Input:
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)
        Y_test : labels (-1 or +1) (n)
    
    Output:
        loglikelihood of each point in X_test (n)
    """
    n, d = X_test.shape
    loglikelihood = np.zeros(n)
    
    # YOUR CODE HERE
    positive = (Y_test == 1)
    negative = (Y_test == -1)
    
    loglikelihood[positive] = X_test[positive]@np.log(posprob) + (1 - X_test[positive])@np.log(1 - posprob)
    loglikelihood[negative] = X_test[negative]@np.log(negprob) + (1 - X_test[negative])@np.log(1 - negprob)

    return loglikelihood

# compute the loglikelihood of the training set
posprob, negprob = naivebayesPXY(X,Y)
loglikelihood(posprob,negprob,X,Y) 




# The following tests check that your implementation of loglikelihood returns the same values as the correct implementation for three different datasets

X, Y = genTrainFeatures(128)
posprobXY, negprobXY = naivebayesPXY_grader(X, Y)

# test if the log likelihood of the training data are all negative
def loglikelihood_testneg():
    ll=loglikelihood(posprob,negprob,X,Y);
    return all(ll<0)

# test if the log likelihood of the training data matches the solution
def loglikelihood_test0():
    ll=loglikelihood(posprob,negprob,X,Y);
    llgrader=loglikelihood_grader(posprob,negprob,X,Y);
    return np.linalg.norm(ll-llgrader)<1e-5

# test if the log likelihood of the training data matches the solution
# (positive points only)
def loglikelihood_test0a():
    ll=loglikelihood(posprob,negprob,X,Y);
    llgrader=loglikelihood_grader(posprob,negprob,X,Y);
    return np.linalg.norm(ll[Y==1]-llgrader[Y==1])<1e-5

# test if the log likelihood of the training data matches the solution
# (negative points only)
def loglikelihood_test0b():
    ll=loglikelihood(posprob,negprob,X,Y);
    llgrader=loglikelihood_grader(posprob,negprob,X,Y);
    return np.linalg.norm(ll[Y==-1]-llgrader[Y==-1])<1e-5


# little toy example with two data points (1 positive, 1 negative)
def loglikelihood_test1():
    x = np.array([[0,1],[1,0]])
    y = np.array([-1,1])
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    loglike = loglikelihood(posprobXY[:2], negprobXY[:2], x, y)
    loglike0 = loglikelihood_grader(posprobXY[:2], negprobXY[:2], x, y)
    test = np.linalg.norm(loglike - loglike0)
    return test < 1e-5

# little toy example with four data points (2 positive, 2 negative)
def loglikelihood_test2():
    x = np.array([[1,0,1,0,1,1], 
        [0,0,1,0,1,1], 
        [1,0,0,1,1,1], 
        [1,1,0,0,1,1]])
    y = np.array([-1,1,1,-1])
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    loglike = loglikelihood(posprobXY[:6], negprobXY[:6], x, y)
    loglike0 = loglikelihood_grader(posprobXY[:6], negprobXY[:6], x, y)
    test = np.linalg.norm(loglike - loglike0)
    return test < 1e-5


# one more toy example with 5 positive and 2 negative points
def loglikelihood_test3():
    x = np.array([[1,1,1,1,1,1], 
        [0,0,1,0,0,0], 
        [1,1,0,1,1,1], 
        [0,1,0,0,0,1], 
        [0,1,1,0,1,1], 
        [1,0,0,0,0,1], 
        [0,1,1,0,1,1]])
    y = np.array([1, 1, 1 ,1,-1,-1, 1])
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    loglike = loglikelihood(posprobXY[:6], negprobXY[:6], x, y)
    loglike0 = loglikelihood_grader(posprobXY[:6], negprobXY[:6], x, y)
    test = np.linalg.norm(loglike - loglike0)
    return test < 1e-5


runtest(loglikelihood_testneg, 'loglikelihood_testneg (all log likelihoods must be negative)')
runtest(loglikelihood_test0, 'loglikelihood_test0 (training data)')
runtest(loglikelihood_test0a, 'loglikelihood_test0a (positive points)')
runtest(loglikelihood_test0b, 'loglikelihood_test0b (negative points)')
runtest(loglikelihood_test1, 'loglikelihood_test1')
runtest(loglikelihood_test2, 'loglikelihood_test2')
runtest(loglikelihood_test3, 'loglikelihood_test3')
