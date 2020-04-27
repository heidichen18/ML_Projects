def onclick_forest(event):
    """
    Visualize forest, including new point
    """
    global xTrain,yTrain,w,b,M,Q,trees,weights
    
    if event.key == 'shift': Q+=10
    else: Q+=1
    Q=min(Q,M)

    classvals = np.unique(yTrain)
        
    # return 300 evenly spaced numbers over this interval
    res=300
    xrange = np.linspace(0, 1,res)
    yrange = np.linspace(0, 1,res)
    
    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # get forest
    
    fun = lambda X:evalboostforest(trees[:Q],X, weights[:Q])        
    # test all of these points on the grid
    testpreds = fun(xTe)
    trerr=np.mean(np.sign(fun(xTrain))==np.sign(yTrain))
    
    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    
    plt.cla()    
    plt.xlim((0,1))
    plt.ylim((0,1))
    # fill in the contours for these predictions
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)
    
    for idx, c in enumerate(classvals):
        plt.scatter(xTrain[yTrain == c,0],xTrain[yTrain == c,1],marker=marker_symbols[idx],color='k')
    plt.show()
    plt.title('# Trees: %i Training Accuracy: %2.2f' % (Q,trerr))
    
        
xTrain=xTrSpiral.copy()/14+0.5
yTrain=yTrSpiral.copy()
yTrain=yTrain.astype(int)

# Hyper-parameters (feel free to play with them)
M=50
alpha=0.05;
depth=5;
trees, weights=GBRT(xTrain, yTrain, M,alpha=alpha,maxdepth=depth)
Q=0;


%matplotlib notebook
fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', onclick_forest) 
print('Click to add a tree.');
plt.title('Click to start boosting on the spiral data.')
visclassifier(lambda X: np.sum(X,1)*0,xTrain,yTrain,newfig=False)
plt.xlim(0,1)
plt.ylim(0,1)
