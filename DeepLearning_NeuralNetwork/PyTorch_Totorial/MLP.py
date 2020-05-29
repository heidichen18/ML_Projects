# Running the following code with generate
# 600 datapoints
# 300 in class 0 
# 300 in class 1
X, y = h.spiraldata(N=300)

h.visualize_2D(X.numpy(), y.numpy())

# create a model class named MLP
# overloads nn.Module 

# This MLP is a simple multilayer perceptron with 3-hidden layers 
# with ReLU transition function
class MLP(nn.Module):
    def __init__(self):
        # call the constructor of the inherited module
        # this line is always necessary
        super().__init__()
        
        # All the layers with trainable parameters have to be 
        # defined in the constructor
        # Create a linear layer that takes input of dimension 2
        # and outputs a vector of dimension 64
        # Similarly, create the consequent layers
        self.fc1 = nn.Linear(in_features=2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        
        # The last layer should output a two dimensional vector
        # where the first dimension corresponds to the network's
        # belief that the input belongs to class 0
        # and similarly, second dimension corresponds to class 1
        self.fc4 = nn.Linear(in_features=16, out_features=2)
        
    def forward(self, x):
        # apply the first linear transformation to input x
        # then apply the ReLU transition to get the output
        # of the first hidden layer o1
        # Similar procedures are done to get o2, o3
        o1 = F.relu(self.fc1(x))
        o2 = F.relu(self.fc2(o1))
        o3 = F.relu(self.fc3(o2))
        
        # We do not apply any transition function to the output
        # of the final hidden layer
        o4 = self.fc4(o3)
        return o4



# Instantiate a model
model = MLP()

# Run forward propagation
logits = model(X)

# Print the output of the model
# This should be (600, 2)
print(logits.shape)
