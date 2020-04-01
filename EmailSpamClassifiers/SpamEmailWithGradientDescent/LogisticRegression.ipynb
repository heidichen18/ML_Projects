def logistic_regression(X, y, max_iter, alpha):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = np.zeros(max_iter)    
    
    for step in range(max_iter):
        wgrad, bgrad=gradient(X, y, w, b)
        w -= np.dot(alpha, wgrad)
        b -= np.dot(alpha, bgrad)
        losses[step] = log_loss(X, y, w, b)
        
    return w, b, losses

weight, b, losses = logistic_regression(features, labels, 1000, 1e-04)
plot(losses)
xlabel('iterations')
ylabel('log_loss')
# your loss should go down :-)



def test_logistic_regression1():

    XUnit = np.array([[-1,1],[-1,0],[0,-1],[-1,2],[1,-2],[1,-1],[1,0],[0,1],[1,-2],[-1,2]])
    YUnit = np.hstack((np.ones(5), -np.ones(5)))

    w1, b1, _ = logistic_regression(XUnit, YUnit, 30000, 5e-5)
    w2, b2, _ = logistic_regression_grader(XUnit, YUnit, 30000, 5e-5)
    return (np.linalg.norm(w1 - w2) < 1e-5) and (np.linalg.norm(b1 - b2) < 1e-5)

def test_logistic_regression2():
    X = np.vstack((np.random.randn(50, 5), np.random.randn(50, 5) + 2))
    Y = np.hstack((np.ones(50), -np.ones(50)))
    max_iter = 300
    alpha = 1e-5
    w1, b1, _ = logistic_regression(X, Y, max_iter, alpha)
    w2, b2, _ = logistic_regression_grader(X, Y, max_iter, alpha)
    return (np.linalg.norm(w1 - w2) < 1e-5) and (np.linalg.norm(b1 - b2) < 1e-5)

def test_logistic_regression3(): # check if losses match predictions
    X = np.vstack((np.random.randn(50, 5), np.random.randn(50, 5) + 2))
    Y = np.hstack((np.ones(50), -np.ones(50)))
    max_iter = 30
    alpha = 1e-5
    w1, b1, losses1 = logistic_regression(X, Y, max_iter, alpha)
    return np.abs(log_loss(X,Y,w1,b1)-losses1[-1])<1e-09

def test_logistic_regression4(): # check if loss decreases
    X = np.vstack((np.random.randn(50, 5), np.random.randn(50, 5) + 2))
    Y = np.hstack((np.ones(50), -np.ones(50)))
    max_iter = 30
    alpha = 1e-5
    w1, b1, losses1 = logistic_regression(X, Y, max_iter, alpha)
    return losses[-1]<losses[0]

runtest(test_logistic_regression1, 'test_logistic_regression1')
runtest(test_logistic_regression2, 'test_logistic_regression2')
runtest(test_logistic_regression3, 'test_logistic_regression3')
runtest(test_logistic_regression4, 'test_logistic_regression4')
