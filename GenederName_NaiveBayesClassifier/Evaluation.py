DIMS = 128
print('Loading data ...')
X,Y = genTrainFeatures(DIMS)
print('Training classifier ...')
pos, neg = naivebayesPY(X, Y)
posprob, negprob = naivebayesPXY(X, Y)
error = np.mean(naivebayes_pred(pos, neg, posprob, negprob, X) != Y)
print('Training error: %.2f%%' % (100 * error))

while True:
    print('Please enter a baby name>')
    yourname = input()
    if len(yourname) < 1:
        break
    xtest = name2features(yourname,B=DIMS,LoadFile=False)
    pred = naivebayes_pred(pos, neg, posprob, negprob, xtest)
    if pred > 0:
        print("%s, I am sure you are a baby boy.\n" % yourname)
    else:
        print("%s, I am sure you are a baby girl.\n" % yourname

)
