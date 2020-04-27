def grid_search(xTr, yTr, xVal, yVal, depths):
    '''
    Input:
        xTr: nxd matrix
        yTr: n vector
        xVal: mxd matrix
        yVal: m vector
        depths: a list of len k
    Return:
        best_depth: the depth that yields that lowest loss on the validation set
        training losses: a list of len k. the i-th entry corresponds to the the training loss
                the tree of depths[i]
        validation_losses: a list of len k. the i-th entry corresponds to the the validation loss
                the tree of depths[i]
    '''
    training_losses = []
    validation_losses = []
    best_depth = None
    
    for i in depths:
        tree = RegressionTree(i)
        tree.fit(xTr, yTr)
        
        training_loss = square_loss(tree.predict(xTr), yTr)
        validation_loss = square_loss(tree.predict(xVal), yVal)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
    
    best_depth = depths[np.argmin(validation_losses)]
    
    return best_depth, training_losses, validation_losses



depths = [1,2,3,4,5]
k = len(depths)
best_depth, training_losses, validation_losses = grid_search(xTr, yTr, xTe, yTe, depths)
best_depth_grader, training_losses_grader, validation_losses_grader = grid_search_grader(xTr, yTr, xTe, yTe, depths)

# Check the length of the training loss
def grid_search_test1():
    return (len(training_losses) == k) 

# Check the length of the validation loss
def grid_search_test2():
    return (len(validation_losses) == k)

# Check the argmin
def grid_search_test3():
    return (best_depth == depths[np.argmin(validation_losses)])

def grid_search_test4():
    return (best_depth == best_depth_grader)

def grid_search_test5():
    return np.linalg.norm(np.array(training_losses) - np.array(training_losses_grader)) < 1e-7

def grid_search_test6():
    return np.linalg.norm(np.array(validation_losses) - np.array(validation_losses_grader)) < 1e-7

runtest(grid_search_test1, 'grid_search_test1')
runtest(grid_search_test2, 'grid_search_test2')
runtest(grid_search_test3, 'grid_search_test3')
runtest(grid_search_test4, 'grid_search_test4')
runtest(grid_search_test5, 'grid_search_test5')
runtest(grid_search_test6, 'grid_search_test6')
