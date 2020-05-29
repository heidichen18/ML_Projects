# Evaluate the model
# Need to go into eval mode
model_v2.eval()

# Do a forward pass and tell PyTorch to not keep track of
# the gradient calculation to save memory
with torch.no_grad():
    logits = model_v2(X)

# make prediction
prediction = torch.argmax(logits, dim=1)

# Calculate the accracuy
print('Accuracy: {:.2f}%'.format(torch.mean((prediction == y).float()).item() * 100))



# visualize the model
h.visclassifier(model_v2, X.numpy(), y.numpy())
