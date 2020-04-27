def generate_kFold(n, k):
    '''
    Input:
        n: number of training examples
        k: number of folds
    Returns:
        kfold_indices: a list of len k. Each entry takes the form
        (training indices, validation indices)
    '''
    assert k >= 2
    kfold_indices = []
    
    indices = np.array(range(n))
    fold_size = n // k
    
    fold_indices = [indices[i*fold_size: (i+1)*fold_size] for i in range(k - 1)]
    fold_indices.append(indices[(k-1)*fold_size:])
    
    
    for i in range(k):
        training_indices = [fold_indices[j] for j in range(k) if j != i]
        validation_indices = fold_indices[i]
        kfold_indices.append((np.concatenate(training_indices), validation_indices))
    
    return kfold_indices

generate_kFold(3,3)




kfold_indices = generate_kFold(1004, 5)

def generate_kFold_test1():
    return len(kfold_indices) == 5 # you should generate 5 folds

def generate_kFold_test2():
    t = [((len(train_indices) + len(test_indices)) == 1004) 
         for (train_indices, test_indices) in kfold_indices]
    return np.all(t) # make sure that both for each fold, the number of examples sum up to 1004

def generate_kFold_test3():
    ratio_test = []
    for (train_indices, validation_indices) in kfold_indices:
        ratio = len(validation_indices) / len(train_indices)
        ratio_test.append((ratio > 0.24 and ratio < 0.26))
    # make sure that both for each fold, the training to validation 
    # examples ratio is in between 0.24 and 0.25
    return np.all(ratio_test) 

def generate_kFold_test4():
    train_indices_set = set() # to keep track of training indices for each fold
    validation_indices_set = set() # to keep track of validation indices for each fold
    for (train_indices, validation_indices) in kfold_indices:
        train_indices_set = train_indices_set.union(set(train_indices))
        validation_indices_set = validation_indices_set.union(set(validation_indices))
    
    # Make sure that you use all the examples in all the training fold and validation fold
    return train_indices_set == set(np.arange(1004)) and validation_indices_set == set(np.arange(1004))


runtest(generate_kFold_test1, 'generate_kFold_test1')
runtest(generate_kFold_test2, 'generate_kFold_test2')
runtest(generate_kFold_test3, 'generate_kFold_test3')
runtest(generate_kFold_test4, 'generate_kFold_test4')
