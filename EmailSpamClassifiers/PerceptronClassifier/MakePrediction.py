def classify_linear(xs,w,b=None):
    """
    function preds=classify_linear(xs,w,b)
    
    Make predictions with a linear classifier
    Input:
    xs : n input vectors of d dimensions (nxd) [could also be a single vector of d dimensions]
    w : weight vector of dimensionality d
    b : bias (scalar)
    
    Output:
    preds: predictions (1xn)
    """    
    w = w.flatten()    
    predictions=np.zeros(xs.shape[0])

    # YOUR CODE HERE
    n,d = xs.shape
    if (b == None):
        b = 0
    for i in range(n):
        predictions[i] = np.dot(xs[i,:],w)+b
        
        predictions = np.sign(predictions)
        
    return predictions





# Run this self-test to check that your linear classifier correctly classifies the points in a linearly separable dataset

def test_linear1():
    xs = np.random.rand(50000,20)-0.5 # draw random data 
    w0 = np.random.rand(20)
    b0 =- 0.1 # with bias -0.1
    ys = classify_linear(xs,w0,b0)
    uniquepredictions=np.unique(ys) # check if predictions are only -1 or 1
    return set(uniquepredictions)==set([-1,1])

def test_linear2():
    xs = np.random.rand(1000,2)-0.5 # draw random data 
    w0 = np.array([0.5,-0.3]) # define a random hyperplane 
    b0 =- 0.1 # with bias -0.1
    ys = np.sign(xs.dot(w0)+b0) # assign labels according to this hyperplane (so you know it is linearly separable)
    return (all(np.sign(ys*classify_linear(xs,w0,b0))==1.0))  # the original hyperplane (w0,b0) should classify all correctly

runtest(test_linear1, 'test_linear1')
runtest(test_linear2, 'test_linear2')
