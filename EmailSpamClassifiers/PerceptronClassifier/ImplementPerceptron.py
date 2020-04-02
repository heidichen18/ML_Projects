def perceptron(xs,ys):
    """
    function w=perceptron(xs,ys);
    
    Implementation of a Perceptron classifier
    Input:
    xs : n input vectors of d dimensions (nxd)
    ys : n labels (-1 or +1)
    
    Output:
    w : weight vector (1xd)
    b : bias term
    """

    n, d = xs.shape     # so we have n input vectors, of d dimensions each
    w = np.zeros(d)
    b = 0.0
    
    # YOUR CODE HERE
    Iter = 100
    i = 0
    
    while (i < Iter):
        misclass = 0
        # Randomize the order in the training data
        for i in np.random.permutation(n):
            if ys[i]*(np.dot(w, xs[i]) + b) <= 0:
                perceptron_update(xs[i], ys[i], w)
                b += ys[i]
                misclass += 1
                
        if (misclass == 0): 
            break
        i += 1    

    return (w, b)
    




# These self tests will check that your perceptron function successfully classifies points in two different linearly separable dataset 

def test_Perceptron1():
    N = 100;
    d = 10;
    x = np.random.rand(N,d)
    w = np.random.rand(1,d)
    y = np.sign(w.dot(x.T))[0]
    w, b = perceptron(x,y)
    preds = classify_linear_grader(x,w,b)
    return np.array_equal(preds.reshape(-1,),y.reshape(-1,))



def test_Perceptron2():
    x = np.array([ [-0.70072, -1.15826],  [-2.23769, -1.42917],  [-1.28357, -3.52909],  [-3.27927, -1.47949],  [-1.98508, -0.65195],  [-1.40251, -1.27096],  [-3.35145,-0.50274],  [-1.37491,-3.74950],  [-3.44509,-2.82399],  [-0.99489,-1.90591],   [0.63155,1.83584],   [2.41051,1.13768],  [-0.19401,0.62158],   [2.08617,4.41117],   [2.20720,1.24066],   [0.32384,3.39487],   [1.44111,1.48273],   [0.59591,0.87830],   [2.96363,3.00412],   [1.70080,1.80916]])
    y = np.array([1]*10 + [-1]*10)
    w, b =perceptron(x,y)
    preds = classify_linear_grader(x,w,b)
    return np.array_equal(preds.reshape(-1,),y.reshape(-1,))

runtest(test_Perceptron1, 'test_Perceptron1')
runtest(test_Perceptron2, 'test_Perceptron2')


