def computenoise(xTe, yTe, OFFSET):
    """
    function noise=computenoise(xTe, OFFSET);

    computes the noise, or square mean of ybar - y, for a set of inputs x
    generated from two standard Normal distributions (one offset by OFFSET in
    both dimensions.)

    INPUT:
    xTe       : nx2 array of n vectors with 2 dimensions
    OFFSET    : The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.

    OUTPUT:
    noise:    : a scalar representing the noise component of the error of xTe
    """
    noise = 0
    
    # YOUR CODE HERE
    ybar = computeybar(xTe, OFFSET)
    noise = np.mean(np.power(ybar - yTe, 2))
    
    return noise
   

def test_noise1():
    OFFSET = 2
    n = 1000
    xTe, yTe = toydata(OFFSET, n) # Generate n datapoints
    noise = computenoise(xTe, yTe, OFFSET)
    
    return np.isscalar(noise) 

def test_noise2():
    OFFSET = 50
    # Create an easy dataset
    # We set sigma=1 and since the mean is far apart,
    # the noise is negligible
    xTe = np.array([
        [49.308783, 49.620651], 
        [1.705462, 1.885418], 
        [ 51.192402, 50.256330],
        [0.205998, -0.089885],
        [50.853083, 51.833237]])  
    yTe = np.array([2, 1, 2, 1, 2])
    noise = computenoise(xTe, yTe, OFFSET)
    return np.isclose(noise,0)

def test_noise3():
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
    
    ybar = computeybar(xTe, OFFSET)
    noise = computenoise(xTe,yTe,OFFSET)
    
    return noise < 0.0002 # make sure the noise is small

runtest(test_noise1, 'test_noise1')
runtest(test_noise2, 'test_noise2')
runtest(test_noise3, 'test_noise3') 
