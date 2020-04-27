def computebias(xTe, depth, Nsmall, NMODELS, OFFSET):
    """
    function bias = computebias(xTe, sigma, lmbda, NSmall, NMODELS, OFFSET);

    computes the bias for data set xTe. 

    The regression tree should be trained using data of size Nsmall and is drawn from toydata with OFFSET 
    

    The "infinite" number of models is estimated as an average over NMODELS. 

    INPUT:
    xTe       | nx2 matrix, of n column-wise input vectors (each 2-dimensional)
    depth     | Depth of the tree 
    NSmall    | Number of points to subsample
    NMODELS   | Number of Models to average over
    OFFSET    | The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.
    OUTPUT:
    bias | a scalar representing the bias of the input data
    """
    noise = 0
    
    # YOUR CODE HERE
    ybar = computeybar(xTe, OFFSET)
    hbar = computehbar(xTe, depth, Nsmall, NMODELS, OFFSET)
    
    bias = np.mean(np.power(hbar-ybar,2))
    
    
    return bias



def test_bias1():
    OFFSET = 2
    depth = 2
    Nsmall = 10
    NMODELS = 10 
    n = 1000
    xTe, yTe = toydata(OFFSET, n)
    bias = computebias(xTe, depth, Nsmall, NMODELS, OFFSET)
    return np.isscalar(bias) # the dimension of hbar should be (n, )

def test_bias2():
    OFFSET = 50
    # Create an easy dataset
    # We set sigma=1 and since the mean is far apart,
    # the noise is negligible
    xTe = np.array([
        [49.308783, 49.620651], 
        [1.705462, 1.885418], 
        [51.192402, 50.256330],
        [0.205998, -0.089885],
        [50.853083, 51.833237]])  
    yTe = np.array([2, 1, 2, 1, 2])
    
    depth = 2
    Nsmall = 10
    NMODELS = 1
    
    # since the mean is far apart, the tree should be able to learn perfectly
    bias = computebias(xTe, depth, Nsmall, NMODELS, OFFSET)
    return np.isclose(bias, 0) # the bias should be close to zero

def test_bias3():
    OFFSET = 3;

    xTe = np.array([
        [0.45864, 0.71552],
        [2.44662, 1.68167],
        [1.00345, 0.15182],
        [-0.10560, -0.48155],
        [3.07264, 3.81535],
        [3.13035, 2.72151],
        [2.25265, 3.78697]])
    yTe = np.array([1, 2, 1, 1, 2, 2, 2])
    
    depth = 3
    Nsmall = 10
    NMODELS = 100
    
    # set the random seed to ensure consistent behavior
    bias = computebias(xTe, depth, Nsmall, NMODELS, OFFSET)
    return np.abs(bias - 0.0017) < 0.001 # the bias should be close to 0.007

runtest(test_bias1, 'test_bias1')
runtest(test_bias2, 'test_bias2')
runtest(test_bias3, 'test_bias3')
