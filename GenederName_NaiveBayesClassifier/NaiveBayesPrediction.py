def naivebayes_pred(pos, neg, posprob, negprob, X_test):
    """
    naivebayes_pred(pos, neg, posprob, negprob, X_test) returns the prediction of each point in X_test
    
    Input:
        pos: class probability for the negative class
        neg: class probability for the positive class
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)
    
    Output:
        prediction of each point in X_test (n)
    """
    n, d = X_test.shape
    
    # YOUR CODE HERE
    ratio1 = loglikelihood_grader(posprob, negprob, X_test, np.ones(n)) - loglikelihood_grader(posprob, negprob, X_test, -np.ones(n))
    ratio2 = np.log(pos) - np.log(neg)
    loglikelihood_ratio = ratio1 + ratio2
    
    prediction = - np.ones(n)
    prediction[loglikelihood_ratio > 0] = 1
    return prediction


# The following tests check that your implementation of naivebayes_pred returns only 1s and -1s (test 1), and that it returns the same predicted values as the correct implementation for three different datasets (tests 2-4)

X,Y = genTrainFeatures_grader(128)
posY, negY = naivebayesPY_grader(X, Y)

# check whether the predictions are +1 or neg 1
def naivebayes_pred_test1():
    preds = naivebayes_pred(posY, negY, posprobXY, negprobXY, X)
    return np.all(np.logical_or(preds == -1 , preds == 1))

def naivebayes_pred_test2():
    naivebayesPXY_grader(X, Y)
    x_test = np.array([[0,1],[1,0]])
    preds = naivebayes_pred_grader(posY, negY, posprobXY[:2], negprobXY[:2], x_test)
    student_preds = naivebayes_pred(posY, negY, posprobXY[:2], negprobXY[:2], x_test)
    acc = analyze_grader("acc", preds, student_preds)
    return np.abs(acc - 1) < 1e-5

def naivebayes_pred_test3():
    x_test = np.array([[1,0,1,0,1,1], 
        [0,0,1,0,1,1], 
        [1,0,0,1,1,1], 
        [1,1,0,0,1,1]])
    naivebayesPXY_grader(X, Y)
    preds = naivebayes_pred_grader(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    student_preds = naivebayes_pred(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    acc = analyze_grader("acc", preds, student_preds)
    return np.abs(acc - 1) < 1e-5

def naivebayes_pred_test4():
    x_test = np.array([[1,1,1,1,1,1], 
        [0,0,1,0,0,0], 
        [1,1,0,1,1,1], 
        [0,1,0,0,0,1], 
        [0,1,1,0,1,1], 
        [1,0,0,0,0,1], 
        [0,1,1,0,1,1]])
    naivebayesPXY_grader(X, Y)
    preds = naivebayes_pred_grader(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    student_preds = naivebayes_pred(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    acc = analyze_grader("acc", preds, student_preds)
    return np.abs(acc - 1) < 1e-5

runtest(naivebayes_pred_test1, 'naivebayes_pred_test1')
runtest(naivebayes_pred_test2, 'naivebayes_pred_test2')
runtest(naivebayes_pred_test3, 'naivebayes_pred_test3')
runtest(naivebayes_pred_test4, 'naivebayes_pred_test4')
