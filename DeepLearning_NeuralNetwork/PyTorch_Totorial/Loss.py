# create the cross entropy loss function
# this loss function takes in logits and labels and return the loss value
# Concretely, loss_fn(logits, y) returns the cross entropy loss between
# the prediction and ground truth
loss_fn = nn.CrossEntropyLoss()


# torch.optim.SGD is the optimizer that PyTorch implemented
# it takes the model's parameter, which you can get by calling
# model.parameters()
# and learning rate lr
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# Num_epochs is the number we are going to go through the dataset
# This is a simple dataset so we are just going to do a vanilla Gradient Descent
# rather than minibatch Stochastic Gradient Descent
num_epochs = 2000

losses = []
for epoch in range(num_epochs):
    # Set the model to training mode
    # This step is always necessary during training 
    model.train()
    
    # PyTorch keeps track of all the gradients generated 
    # throughout the whole training procedure
    # But we only need the gradient of the current time step
    # so we need to zero-out the gradient buffer
    optimizer.zero_grad()
    
    # Do forward propagation
    logits = model(X)
    
    # Calculate the loss 
    loss = loss_fn(logits, y)
    
    # Do backward propagation
    loss.backward()
    
    # Update the model parameters 
    optimizer.step()
    
    # Get the prediction
    prediction = torch.argmax(logits, dim=1)
    
    # PyTorch supports elements elementwise comparison
    acc = torch.mean((prediction == y).float())
    
    losses.append(loss.item())
    
    # print the loss value every 50 epochs
    if (epoch + 1) % 50 == 0:
        print('Epoch [{}/ {}], Loss: {:.4f} '.format(epoch + 1, num_epochs, losses[-1]))



# Plot the training loss
plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.show()
