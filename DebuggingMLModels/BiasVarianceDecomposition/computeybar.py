def computeybar(xTe, OFFSET):
    """
    function [ybar]=computeybar(xTe, OFFSET);

    computes the expected label 'ybar' for a set of inputs x
    generated from two standard Normal distributions (one offset by OFFSET in
    both dimensions.)

    INPUT:
    xTe       : nx2 array of n vectors with 2 dimensions
    OFFSET    : The OFFSET passed into the toyData function. The difference in the
                mu of labels class1 and class2 for toyData.

    OUTPUT:
    ybar : a nx1 vector of the expected labels for vectors xTe
    noise: 
    """
    n, d = xTe.shape
    ybar = np.zeros(n)
    
    # Feel free to use the following function to compute p(x|y)
    # By default, mean is 0 and std. deviation is 1.
    normpdf = lambda x, mu, sigma: np.exp(-0.5 * np.power((x - mu) / sigma, 2)) / (np.sqrt(2 * np.pi) * sigma)
    
    # YOUR CODE HERE
    c1 = normpdf(xTe, 0, 1)
    c2 = normpdf(xTe, OFFSET, 1)
    
    c1 = np.multiply(c1[:,0], c1[:,1])
    c2 = np.multiply(c2[:,0], c2[:,1])
    
    ybar = (c1+2*c2) / (c1+c2)
    return ybar


def test_ybar1():
    OFFSET = 2
    n = 1000
    xTe, yTe = toydata(OFFSET, n) # Generate n datapoints
    ybar = computeybar(xTe, OFFSET)
    
    return ybar.shape == (n, ) # the output of your ybar should be a n dimensional array

def test_ybar2():
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
    
    ybar = computeybar(xTe, OFFSET)
    return np.isclose(np.mean(np.power(yTe - ybar, 2)), 0)

def test_ybar3():
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
    
    return np.mean(np.power(yTe - ybar, 2)) < 0.0002 # make sure the noise is small

runtest(test_ybar1, 'test_ybar1')
runtest(test_ybar2, 'test_ybar2')
runtest(test_ybar3, 'test_ybar3')
