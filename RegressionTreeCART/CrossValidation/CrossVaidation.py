def cross_validation(xTr, yTr, depths, indices):
    '''
    Input:
        xTr: nxd matrix (training data)
        yTr: n vector (training data)
        depths: a list (of length l) depths to be tried out
        indices: indices from generate_kFold
    Returns:
        best_depth: the best parameter 
        training losses: a list of lenth l. the i-th entry corresponds to the the average training loss
                the tree of depths[i]
        validation_losses: a list of length l. the i-th entry corresponds to the the average validation loss
                the tree of depths[i] 
    '''
    training_losses = []
    validation_losses = []
    best_depth = None
    
    for train_indices, validation_indices in indices:
        xtrain, ytrain = xTr[train_indices], yTr[train_indices]
        xval, yval = xTr[validation_indices], yTr[validation_indices]
        
        _, training_loss, validation_loss = grid_search(xtrain, ytrain, xval, yval, depths)
        
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
    
    training_losses = np.mean(training_losses, axis=0)
    validation_losses = np.mean(validation_losses, axis=0)
    best_depth = depths[np.argmin(validation_losses)]
    
    return best_depth, training_losses, validation_losses


depths = [1,2,3,4]
k = len(depths)

# generate indices
# the same indices will be used to cross check your solution and ours
indices = generate_kFold(len(xTr), 5)
best_depth, training_losses, validation_losses = cross_validation(xTr, yTr, depths, indices)
best_depth_grader, training_losses_grader, validation_losses_grader = cross_validation_grader(xTr, yTr, depths, indices)

# Check the length of the training loss
def cross_validation_test1():
    return (len(training_losses) == k) 

# Check the length of the validation loss
def cross_validation_test2():
    return (len(validation_losses) == k)

# Check the argmin
def cross_validation_test3():
    return (best_depth == depths[np.argmin(validation_losses)])

def cross_validation_test4():
    return (best_depth == best_depth_grader)

def cross_validation_test5():
    return np.linalg.norm(np.array(training_losses) - np.array(training_losses_grader)) < 1e-7

def cross_validation_test6():
    return np.linalg.norm(np.array(validation_losses) - np.array(validation_losses_grader)) < 1e-7

runtest(cross_validation_test1, 'cross_validation_test1')
runtest(cross_validation_test2, 'cross_validation_test2')
runtest(cross_validation_test3, 'cross_validation_test3')
runtest(cross_validation_test4, 'cross_validation_test4')
runtest(cross_validation_test5, 'cross_validation_test5')
runtest(cross_validation_test6, 'cross_validation_test6')
