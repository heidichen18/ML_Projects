# Create a model
model = DAN(len(vocab), embedding_size=32)

if gpu_available:
    model = model.cuda()


# Create optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5)




