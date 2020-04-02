def mode(y):
   if len(y) == 1:
       return y[0]
   else:
       counts = {}
       for i in range(len(y)):
           if y[i] in counts:
               counts[y[i]] +=1
           else:
               counts[y[i]] =1
       m = max(counts, key=counts.get)
       c = counts[m]
       del counts[m]
       if len(counts) == 0:
           return m
       else:
           second_highest = max(counts, key=counts.get)
 
           if counts[second_highest] == c:
               return mode(y[:(len(y)-1)])
           else:
               return m
               
 def knnclassifier(xTr,yTr,xTe,k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);
    
    k-nn classifier 
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    
    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    # fix array shapes
    yTr = yTr.flatten()

    # YOUR CODE HERE
    tup = findknn(xTr, xTe, k)
    I = np.transpose(tup[0])
    D = np.transpose(tup[1])
    preds = []
    for i in range(len(xTe)):
       inds = I[i]
       y = np.ndarray.flatten(yTr[inds])
       m = mode(y)
       preds.append(m)

    preds = np.ndarray.flatten(np.array(preds))
    return preds
