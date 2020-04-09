def evaltree(root,xTe,index=[]):
    """Evaluates xTe using decision tree root.
    
    Input:
        root: TreeNode decision tree
        xTe:  n x d matrix of data points
    
    Output:
        pred: n-dimensional vector of predictions
    """
    n = xTe.shape[0]
    pred = np.zeros(n)

    
    if len(index)==0: 
        index=np.ones(n)==1 

    if root.left is None and root.right is None:
        pred = np.ones(sum(index))*root.prediction
        return pred
            
    assert root.left is not None and root.right is not None

    feature = root.feature
    cutoff = root.cut

    indexLeft = index & (xTe[:,feature] <= cutoff)
    
    if root.left.left is None and root.left.right is None:
        pred[indexLeft]=root.left.prediction
    else:
        pred[indexLeft]=evaltree(root.left, xTe,indexLeft) 

    indexRight = index & (xTe[:,feature]  > cutoff)
    if root.right.left is None and root.right.right is None:
        pred[indexRight]=root.right.prediction
    else:
        pred[indexRight]=evaltree(root.right,xTe,indexRight)
    return(pred[index])
    
    return evaltree(root,xTe)


# The following tests check that your implementation of evaltree returns the correct predictions for two sample trees

t0 = time.time()
root = cart(xTrIon, yTrIon)
t1 = time.time()

tr_err   = np.mean((evaltree(root,xTrIon) - yTrIon)**2)
te_err   = np.mean((evaltree(root,xTeIon) - yTeIon)**2)

print("Elapsed time: %.2f seconds" % (t1-t0))
print("Training RMSE : %.2f" % tr_err)
print("Testing  RMSE : %.2f \n" % te_err)

#test case 1
def evaltree_test1():
    t = cart(xor4,yor4)
    xor4te = xor4 + (np.sign(xor4 - .5) * .1)
    inds = np.arange(16)
    np.random.shuffle(inds)
    # Check that shuffling and expanding the data doesn't affect the predictions
    return np.all(np.isclose(evaltree(t, xor4te[inds,:]), yor4[inds]))

#test case 2
def evaltree_test2():
    a = TreeNode(None, None, None, None, 1)
    b = TreeNode(None, None, None, None, -1)
    c = TreeNode(None, None, None, None, 0)
    d = TreeNode(None, None, None, None, -1)
    e = TreeNode(None, None, None, None, -1)
    x = TreeNode(a, b, 0, 10, 0)
    y = TreeNode(x, c, 0, 20, 0)
    z = TreeNode(d, e, 0, 40, 0)
    t = TreeNode(y, z, 0, 30, 0)
    # Check that the custom tree evaluates correctly
    return np.all(np.isclose(
            evaltree(t, np.array([[45, 35, 25, 15, 5]]).T),
            np.array([-1, -1, 0, -1, 1])))

runtest(evaltree_test1,'evaltree_test1')
runtest(evaltree_test2,'evaltree_test2')
