# Start Training
num_epochs = 1000

model.train()
for epoch in range(num_epochs):
    
    # Define the following variables to keep track of the running losses and accuracies
    running_loss = 0.0
    running_acc = 0.0
    count = 0
    
    for i, (X, y) in enumerate(trainloader):
        
        # use gpu if necessary
        if gpu_available:
            X = X.cuda()
            y = y.cuda()
        
        # clear the gradient buffer
        optimizer.zero_grad()
        
        # Do forward propagation to get the model's belief
        logits = model(X)
        
        # Compute the loss
        loss = loss_fn(logits, y)
        
        # Run a backward propagation to get the gradient
        loss.backward()
        
        # Update the model's parameter
        optimizer.step()
        
        # Get the model's prediction 
        pred = torch.argmax(logits,dim=1)
        
        # Update the running statistics
        running_acc += torch.sum((pred == y).float()).item()
        running_loss += loss.item()
        count += X.size(0)
        
    # print the running statistics after training for 100 epochs
    if (epoch + 1) % 100 == 0:
        print('Epoch [{} / {}] Average Training Accuracy: {:4f}'.format(epoch + 1, num_epochs, running_acc / count))
        print('Epoch [{} / {}] Average Training loss: {:4f}'.format(epoch + 1, num_epochs, running_loss / len(trainloader)))



