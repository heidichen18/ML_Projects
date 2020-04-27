def GBRT(xTr, yTr, m, maxdepth=4, alpha=0.1):
    """Creates GBRT.
    
    Input:
        xTr:      n x d matrix of data points
        yTr:      n-dimensional vector of labels
        m:        number of trees in the forest
        maxdepth: maximum depth of tree
        alpha:    learning rate for the GBRT
        
        
    Output:
        trees: list of decision trees of length m
        weights: weights of each tree
    """
    
    n, d = xTr.shape
    trees = []
    weights = []
    
    # Make a copy of the ground truth label
    # this will be the initial ground truth for our GBRT
    # This should be updated for each iteration
    t = np.copy(yTr)
    
    # YOUR CODE HERE
    for i in range(m):
        tree = RegressionTree(depth=maxdepth)
        tree.fit(xTr, t)
        trees.append(tree)
        weights.append(alpha)
        
        # update t
        predictedH = evalboostforest(trees, xTr, weights)
        t = yTr - predictedH
        
    
    return trees, weights




trees, weights = GBRT(xTrSpiral,yTrSpiral, 50)


def GBRT_test1():
    m = 40
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    trees, weights = GBRT(x, y, m, alpha=0.1)
    return len(trees) == m and len(weights) == m # make sure there are m trees in the forest

def GBRT_test2():
    m = 20
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    max_depth = 4
    trees, weights = GBRT(x, y, m, max_depth)
    depths_forest = np.array([tree.depth for tree in trees]) # Get the depth of all trees in the forest
    return np.all(depths_forest == max_depth) # make sure that the max depth of all the trees is correct

def GBRT_test3():
    m = 4
    xTrSpiral,yTrSpiral,_,_= spiraldata(150)
    max_depth = 4
    trees, weights = GBRT(xTrSpiral, yTrSpiral, m, max_depth, 1) # Create a gradient boosted forest with 4 trees
    
    errs = [] 
    for i in range(m):
        predH = evalboostforest(trees[:i+1], xTrSpiral, weights[:i+1]) # calculate the prediction of the first i-th tree
        err = np.mean(np.sign(predH) != yTrSpiral) # calculate the error of the first i-th tree
        errs.append(err) # keep track of the error
    
    # your errs should be decreasing, i.e., the different between two subsequent errors is <= 0
    return np.all(np.diff(errs) <= 0) 
    
runtest(GBRT_test1, 'GBRT_test1')
runtest(GBRT_test2, 'GBRT_test2')
runtest(GBRT_test3, 'GBRT_test3')
