def findknn(xTr,xTe,k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);
    
    Finds the k nearest neighbors of xTe in xTr.
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """

    # YOUR CODE HERE
    if k > len(xTr):
       k = len(xTr)
       
    D=l2distance(xTe, xTr)
    (m,n) = D.shape
   
    indices = []
    dists = []
    for i in range(m):
       smallest_indices = np.argsort(D[i])
       ind = smallest_indices[:k]
       dis = D[i,smallest_indices[:k]]
       indices.append(ind)
       dists.append(dis)
 
    indices = np.transpose(np.array(indices))
    dists = np.transpose(np.array(dists))
    return indices, dists
