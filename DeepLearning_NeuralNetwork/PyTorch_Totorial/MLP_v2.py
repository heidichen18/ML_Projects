class MLP_v2(nn.Module):
    def __init__(self):
        # call the constructor of the inherited module
        # this line is always necessary
        super().__init__()
        
        # TODO: fill in the blank with the right layer
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        
    def forward(self, x):
        o1 = F.relu(self.fc1(x))
        o2 = F.relu(self.fc2(o1))
        o3 = self.fc3(o2)
        return o3

model_v2 = MLP_v2()

# If you create the right model, you should be able to pass the following test

# Checking the size of the parameters in the first layer
fc1_parameters = list(model_v2.fc1.parameters())

# the weight should have size (256, 2)
assert fc1_parameters[0].shape == (256, 2)
# the bias should have size (256, )
assert fc1_parameters[1].shape == (256, )

# Checking the size of the parameters in the second layer
fc2_parameters = list(model_v2.fc2.parameters())
assert fc2_parameters[0].shape == (128, 256)
assert fc2_parameters[1].shape == (128, )

# Checking the size of the parameters in the last layer
fc3_parameters = list(model_v2.fc3.parameters())
assert fc3_parameters[0].shape == (2, 128)
assert fc3_parameters[1].shape == (2, )


# creating a new optimizer
# Feel free to play around with the learning rate
optimizer_v2 = torch.optim.SGD(model_v2.parameters(), lr=0.1)


