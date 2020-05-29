# Train the model
num_epochs = 2000

losses = []
for epoch in range(num_epochs):
    # Set the model to training mode
    # This step is always necessary during training 
    model_v2.train()
    
    # PyTorch keeps track of all the gradients generated 
    # throughout the whole training procedure
    # But we only need the gradient of the current time step
    # so we need to zero-out the gradient buffer
    optimizer_v2.zero_grad()
    
    # Do forward propagation
    logits = model_v2(X)
    
    # Calculate the loss 
    loss = loss_fn(logits, y)
    
    # Do backward propagation
    loss.backward()
    
    # Update the model parameters 
    optimizer_v2.step()
    
    # Get the prediction
    prediction = torch.argmax(logits, dim=1)
    
    # PyTorch supports elements elementwise comparison
    acc = torch.mean((prediction == y).float())
    
    losses.append(loss.item())
    
    # print the loss value every 50 epochs
    if (epoch + 1) % 50 == 0:
        print('Epoch [{}/ {}], Loss: {:.4f} '.format(epoch + 1, num_epochs, losses[-1]))



