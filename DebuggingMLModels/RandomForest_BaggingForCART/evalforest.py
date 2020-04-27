def evalforest(trees, X):
    """Evaluates X using trees.
    
    Input:
        trees:  list of TreeNode decision trees of length m
        X:      n x d matrix of data points
        alphas: m-dimensional weight vector
        
    Output:
        pred: n-dimensional vector of predictions
    """
    m = len(trees)
    n,d = X.shape
    
    pred = np.zeros(n)
    
    # YOUR CODE HERE
    alphas = 1/m
    for i in range(m):
        score = trees[i].predict(X)
        pred += alphas * score
    
    return pred


def evalforest_test1():
    m = 200
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    trees = forest(x, y, m)
    
    preds = evalforest(trees, x)
    return preds.shape == y.shape

def evalforest_test2():
    m = 200
    x = np.ones(10).reshape((10, 1))
    y = np.ones(10)
    max_depth = 0
    
    # Create a forest with trees depth 0
    # Since the data are all ones, each tree will return 1 as prediction
    trees = forest(x, y, m, 0) 
    pred = evalforest(trees, np.ones((1, 1)))[0]
    return np.isclose(pred,1) # the prediction should be equal to the sum of weights
    
def bagging_test1():
    m = 50
    xTr = np.random.rand(500,3) - 0.5
    yTr = np.sign(xTr[:,0] * xTr[:,1] * xTr[:,2]) # XOR Classification
    xTe = np.random.rand(50,3) - 0.5
    yTe = np.sign(xTe[:,0] * xTe[:,1] * xTe[:,2])

    tree = RegressionTree(depth=5)
    tree.fit(xTr, yTr)
    oneacc = np.sum(np.sign(tree.predict(xTe)) == yTe)

    trees = forest(xTr, yTr, m, maxdepth=5)
    multiacc = np.sum(np.sign(evalforest(trees, xTe)) == yTe)

    # Check that bagging yields improvement - or doesn't get too much worse
    return multiacc * 1.1 >= oneacc

def bagging_test2():
    m = 50
    xTr = (np.random.rand(500,3) - 0.5) * 4
    yTr = xTr[:,0] * xTr[:,1] * xTr[:,2] # XOR Regression
    xTe = (np.random.rand(50,3) - 0.5) * 4
    yTe = xTe[:,0] * xTe[:,1] * xTe[:,2]
    
    np.random.seed(1)
    tree = RegressionTree(depth=3)
    tree.fit(xTr, yTr)
    oneerr = np.sum(np.sqrt((tree.predict(xTe) - yTe) ** 2))

    trees = forest(xTr, yTr, m, maxdepth=3)
    multierr = np.sum(np.sqrt((evalforest(trees, xTe) - yTe) ** 2))

    # Check that bagging yields improvement - or doesn't get too much worse
    return multierr <= oneerr * 1.5

runtest(evalforest_test1, 'evalforest_test1')
runtest(evalforest_test2, 'evalforest_test2')
runtest(bagging_test1, 'bagging_test1')
runtest(bagging_test2, 'bagging_test2')
