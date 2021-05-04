# First load data
# Calling load_data return the training review, test review and vocabulary
# review_train and review_test are stored as pandas dataframe
# vocabulary is dictionary with key-value pairs (word, index)
# vocabulary[word] = index
# We will use this vocabulary to construct bag-of-word (bow) features
review_train, review_test, vocab = load_data()

# label 0 == Negative reviews
# label 1 == Positive reviews
label_meaning = ['Negative', 'Positive']

print('Number of Training Reviews: ', review_train.shape[0])
print('Number of Test Reviews: ', review_test.shape[0])
print('Number of Words in the Vocabulary: ', len(vocab))


# print some training review
print('A Positive Training Review: ', review_train.iloc[0]['review'])
print('A Negative Training Review: ', review_train.iloc[-1]['review'])


# Create a simple vocabulary
simple_vocab = {'learn': 0, 'machine': 1, 'learning': 2, 'teach': 3}

# Create a simple sentence that will be converted into bag of words features
simple_sentence = ' I learn machine learning to teach machine how to learn.'

# Create a featurizer by passing in the vocabulary
simple_featurizer = generate_featurizer(simple_vocab)

# Call simple_featurizer.transform to transform the sentence to its bag of word features
simple_featurizer.transform([simple_sentence]).toarray()

# You should get array([[2, 2, 1, 1]]) as output.
# This means that the sentence has 2 occurences of 'learn', 2 occurences of 'machine', 
# 1 occurence of 'learning' and 1 occurence of 'teach'


# Use the given vocab to generate bag of word featurizer
# See the next cell for an example
bow_featurizer = generate_featurizer(vocab)


# convert the reviews to bow representation and torch Tensor
X_train = torch.Tensor(bow_featurizer.transform(review_train['review'].values).toarray())
y_train = torch.LongTensor(review_train['label'].values.flatten())

X_test = torch.Tensor(bow_featurizer.transform(review_test['review'].values).toarray())
y_test = torch.LongTensor(review_test['label'].values.flatten())


# Generate PyTorch Datasets 
trainset = torch.utils.data.TensorDataset(X_train, y_train)
testset = torch.utils.data.TensorDataset(X_test, y_test)

# Generate PyTorch Dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, drop_last=False)
