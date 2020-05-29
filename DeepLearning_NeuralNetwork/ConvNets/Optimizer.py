#TODO: Define the optimizer
optimizer = None

# YOUR CODE HERE
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def optimizer_test1():
    return isinstance(optimizer, torch.optim.SGD)

runtest(optimizer_test1, 'optimizer_test1')
