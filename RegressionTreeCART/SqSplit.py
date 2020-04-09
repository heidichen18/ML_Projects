def sqsplit(xTr, yTr):
    """Finds the best feature, cut value, and loss value.
    
    Input:
        xTr:     n x d matrix of data points
        yTr:     n-dimensional vector of labels
    
    Output:
        feature:  index of the best cut's feature
        cut:      cut-value of the best cut
        bestloss: loss of the best cut
    """
    N,D = xTr.shape
    assert D > 0 # must have at least one dimension
    assert N > 1 # must have at least two samples
    
    bestloss = np.inf
    feature = np.inf
    cut = np.inf

    for d in range(D):
        sort = xTr[:, d].argsort() # sort data
        SortX = xTr[sort, d] # sort feature value
        SortY = yTr[sort]    # sort label
        
        # Get index where to split
        diff = np.diff(SortX, axis=0)
        #print(diff)
        debug = np.isclose(diff, 0, atol=1e-12)
        #print(debug)
        condition = np.logical_not(debug)
        ind = np.asarray(condition).nonzero()[0]
        
        
        for i in ind:
            LossLeft = sqimpurity(SortY[:i + 1]) 
            LossRight = sqimpurity(SortY[i + 1:]) 
            
            loss = LossLeft + LossRight
            
            if loss < bestloss: # best loss so stop split
                bestloss = loss
                feature = d 
                cut = (SortX[i]+SortX[i+1])/2

    
    return feature, cut, bestloss


# The tests below check that your sqsplit function returns the correct values for several different input datasets

t0 = time.time()
fid, cut, loss = sqsplit(xTrIon,yTrIon)
t1 = time.time()

print('Elapsed time: {:0.2f} seconds'.format(t1-t0))
print("The best split is on feature 2 on value 0.304")
print("Your tree split on feature %i on value: %2.3f \n" % (fid,cut))

def sqsplit_test1():
    a = np.isclose(sqsplit(xor4, yor4)[2] / len(yor4), .25)
    b = np.isclose(sqsplit(xor3, yor3)[2] / len(yor3), .25)
    c = np.isclose(sqsplit(xor2, yor2)[2] / len(yor2), .25)
    return a and b and c

def sqsplit_test2():
    x = np.array(range(1000)).reshape(-1,1)
    y = np.hstack([np.ones(500),-1*np.ones(500)]).T
    _, cut, _ = sqsplit(x, y)
    return cut <= 500 or cut >= 499

def sqsplit_test3():
    fid, cut, loss = sqsplit(xor5,yor5)
    # cut should be 0.5 but 0 is also accepted
    return fid == 0 and (cut >= 0 or cut <= 1) and np.isclose(loss / len(yor5), 2/3)

runtest(sqsplit_test1,'sqsplit_test1')
runtest(sqsplit_test2,'sqsplit_test2')
runtest(sqsplit_test3,'sqsplit_test3')
