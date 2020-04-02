def CalAccuracy:
    print("Face Recognition: (1-nn)")
    xTr,yTr,xTe,yTe=loaddata("faces.mat") # load the data
    t0 = time.time()
    preds = knnclassifier(xTr,yTr,xTe,1)
    result=accuracy(yTe,preds)
    t1 = time.time()
    print("You obtained %.2f%% classification acccuracy in %.4f seconds\n" % (result*100.0,t1-t0))
