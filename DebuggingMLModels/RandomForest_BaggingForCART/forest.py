def forest(xTr, yTr, m, maxdepth=np.inf):
    """Creates a random forest.
    
    Input:
        xTr:      n x d matrix of data points
        yTr:      n-dimensional vector of labels
        m:        number of trees in the forest
        maxdepth: maximum depth of tree
        
    Output:
        trees: list of decision trees of length m
    """
    
    n, d = xTr.shape
    trees = []
    
    # YOUR CODE HERE
    for i in range(m):
        indices = np.random.choice(n,n)
        tree = RegressionTree(depth=maxdepth)
        tree.fit(xTr[indices,:], yTr[indices])
        trees.append(tree)
    
    return trees


def forest_test1():
    m = 20
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    trees = forest(x, y, m)
    return len(trees) == m # make sure there are m trees in the forest

def forest_test2():
    m = 20
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    max_depth = 4
    trees = forest(x, y, m, max_depth)
    depths_forest = np.array([tree.depth for tree in trees]) # Get the depth of all trees in the forest
    return np.all(depths_forest == max_depth) # make sure that the max depth of all the trees is correct


def forest_test3():
    s = set()

    def DFScollect(tree):
        # Do Depth first search to all prediction in the tree
        if tree.left is None and tree.right is None:
            s.add(tree.prediction)
        else:
            DFScollect(tree.right)
            DFScollect(tree.left)

    m = 200
    x = np.arange(100).reshape((100, 1))
    y = np.arange(100)
    trees = forest(x, y, m);

    lens = np.zeros(m)

    for i in range(m):
        s.clear()
        DFScollect(trees[i].root)
        lens[i] = len(s)

    # Check that about 63% of data is represented in each random sample
    return abs(np.mean(lens) - 100 * (1 - 1 / np.exp(1))) < 2

runtest(forest_test1, 'forest_test1')
runtest(forest_test2, 'forest_test2')
runtest(forest_test3, 'forest_test3')
