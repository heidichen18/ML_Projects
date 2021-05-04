# Create a Deep Averaging network model class
# embedding_size is the size of the word_embedding we are going to learn
class DAN(nn.Module):
    def __init__(self, vocab_size, embedding_size=32):
        super().__init__()
        
        # Create a word-embedding of dimension embedding_size
        # self.embeds is now the matrix E, where each column corresponds to the embedding of a word
        self.embeds = torch.nn.Parameter(torch.randn(vocab_size, embedding_size))
        self.embeds.requires_grad_(True)       
        # add a final linear layer that computes the 2d output from the averaged word embedding
        self.fc = nn.Linear(embedding_size, 2) 
        
    def average(self, x):
        '''
        This function takes in multiple inputs, stored in one tensor x. Each input is a bag of word representation of reviews. 
        For each review, it retrieves the word embedding of each word in the review and averages them (weighted by the corresponding
        entry in x). 
        
        Input: 
            x: nxd torch Tensor where each row corresponds to bag of word representation of a review
        
        Output:
            n x (embedding_size) torch Tensor for the averaged reivew 
        '''
        
        # YOUR CODE HERE
        total = torch.sum(x, dim=1, keepdim=True)
        matmul = torch.matmul(x, self.embeds)
        emb = matmul / total
        return emb
          
    def forward(self, x):
        '''
        This function takes in a bag of word representation of reviews. It calls the self.average to get the
        averaged review and pass it through the linear layer to produce the model's belief.
        
        Input: 
            x: nxd torch Tensor where each row corresponds to bag of word representation of reviews
        
        Output:
            nx2 torch Tensor that corresponds to model belief of the input. For instance, output[0][0] is
            is the model belief that the 1st review is negative
        '''
        review_averaged = self.average(x)
        
        out = None
        # YOUR CODE HERE
        out = self.fc(review_averaged)
        
        return out



def average_test1(): # check the output dinemsions of the average function
    n = 10 # number of reviews
    vocab_size = 5 # vocab size
    embedding_size = 32 # embedding size
    model = DAN(vocab_size, embedding_size)
    X = torch.rand(n, vocab_size)    
    output_size = model.average(X).shape    
    # the output of your forward function should be nx2
    return output_size[0] == n and output_size[1] == embedding_size

def average_test2():
    n = 10 # number of reviews
    vocab_size = 3 # vocab size
    embedding_size = 5 # embedding size
    model = DAN(vocab_size, embedding_size)
    
    # generate a simple input
    X = torch.FloatTensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ])
    
    # Get the averaged reviews
    averaged_reviews = model.average(X)
    
    # Given the input, we know that the first 3 rows corresponds to the first three words
    # The last row should be the average of the three words
    # The diff between the last row and the average of the first three rows should be small
    diff = torch.sum((torch.mean(averaged_reviews[:3], dim=0) - averaged_reviews[3]) ** 2).item()
    
    return diff < 1e-5

def average_test3():
    n = 10 # number of reviews
    vocab_size = 3 # vocab size
    embedding_size = 5 # embedding size
    model = DAN(vocab_size, embedding_size)
    
    # generate a simple input
    X = torch.FloatTensor([
        [1, 1, 1],
        [2, 2, 2]
    ])
    
    # Get the averaged reviews
    averaged_reviews = model.average(X)
    
    # Since the 2nd review is a multiple of the first,
    # The two averaged review should be the same
    diff = torch.sum((averaged_reviews[0] - averaged_reviews[1])**2).item()
    
    return diff < 1e-5

def forward_test1():
    n = 10 # number of reviews
    vocab_size = 5 # vocab size
    embedding_size = 32 # embedding size
    model = DAN(vocab_size, embedding_size)
    
    # call the forward function
    X = torch.rand(n, vocab_size)
    
    output_size = model(X).shape
    
    # the output of your forward function should be nx2
    return output_size[0] == n and output_size[1] == 2

def forward_test2():
    n = 10 # number of reviews
    vocab_size = 5 # vocab size
    embedding_size = 32 # embedding size
    model = DAN(vocab_size, embedding_size)
    X = torch.rand(n, vocab_size)
    
    logits = model(X) # get the output of your forward pass
    
    averaged_reviews = model.average(X) # get the intermediate averaged review
    logits2 = model.fc(averaged_reviews) # get the model belief using your intermediate average reviews
    
    return torch.sum((logits - logits2)**2).item() < 1e-5 # Check whether your forward pass is implemented correctly

runtest(average_test1, 'average_test1')
runtest(average_test2, 'average_test2')
runtest(average_test3, 'average_test3')
runtest(forward_test1, 'forward_test1')
runtest(forward_test2, 'forward_test2')
