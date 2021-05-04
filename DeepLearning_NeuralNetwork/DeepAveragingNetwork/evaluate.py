# Evaluate the model
model.eval()

# Define the following variable to keep track of the accuracy
running_acc = 0.0
count = 0.0

for (X, y) in testloader:
    # Use gpu if available
    if gpu_available:
        X = X.cuda()
        y = y.cuda()

    # Do a forward pass and tell PyTorch that no gradient is necessary to save memory
    with torch.no_grad():
        logits = model(X)
    
    # Calculate the prediction
    pred = torch.argmax(logits,dim=1)
    
    # Update the running stats
    running_acc += torch.sum((pred == y).float()).item()
    count += X.size(0)

print('Your Test Accuracy is {:.4f}'. format(running_acc / count))



target = torch.randint(high=len(testset), size=(1,)).item()

review_target, label_target = review_test.iloc[target]
if gpu_available:
    bog_target = testset[target][0].unsqueeze(0).cuda()
else:
    bog_target = testset[target][0].unsqueeze(0)


model.eval()
with torch.no_grad():
    logits_target = model(bog_target)

pred = torch.argmax(logits_target, dim=1)
probability=torch.exp(logits_target.squeeze())/torch.sum(torch.exp(logits_target.squeeze()))

print('Review: ', review_target)
print('Ground Truth: ', label_meaning[int(label_target)])
print('Prediction: %s (Certainty %2.2f%%)' % (label_meaning[pred.item()],100.0*probability[pred.item()]))
