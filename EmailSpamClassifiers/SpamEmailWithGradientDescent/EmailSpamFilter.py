import pandas as pd
import dask
import dask.bag
from dask.diagnostics import ProgressBar


train_url = 's3://codio/CIS530/CIS533/data_train'
test_url = 's3://codio/CIS530/CIS533/data_test'


# tokenize the email and hashes the symbols into a vector
def extract_features_naive(email, B):
    # initialize all-zeros feature vector
    v = np.zeros(B)
    email = ' '.join(email)
    # breaks for non-ascii characters
    tokens = email.split()
    for token in tokens:
        v[hash(token) % B] = 1
    return v

def load_spam_data(extract_features, B=512, url=train_url):
    '''
    INPUT:
    extractfeatures : function to extract features
    B               : dimensionality of feature space
    path            : the path of folder to be processed
    
    OUTPUT:
    X, Y
    '''
    
    all_emails = pd.read_csv(url+'/index', header=None).values.flatten()
    
    xs = np.zeros((len(all_emails), B))
    ys = np.zeros(len(all_emails))
    
    labels = [k.split()[0] for k in all_emails]
    paths = [url+'/'+k.split()[1] for k in all_emails]

    ProgressBar().register()
    dask.config.set(scheduler='threads', num_workers=50)
    bag = dask.bag.read_text(paths, storage_options={'anon':True})
    contents = dask.bag.compute(*bag.to_delayed())
    for i, email in enumerate(contents):
        # make labels +1 for "spam" and -1 for "ham"
        ys[i] = (labels[i] == 'spam') * 2 - 1
        xs[i, :] = extract_features(email, B)
    print('Loaded %d input emails.' % len(ys))
    return xs, ys

Xspam, Yspam = load_spam_data(extract_features_naive)
Xspam.shape


# Split data into training (xTr and yTr) 
# and testing (xTv and yTv)
n, d = Xspam.shape
# Allocate 80% of the data for training and 20% for testing
cutoff = int(np.ceil(0.8 * n))
# indices of training samples
xTr = Xspam[:cutoff,:]
yTr = Yspam[:cutoff]
# indices of testing samples
xTv = Xspam[cutoff:]
yTv = Yspam[cutoff:]


# Training and Evaluation
max_iter = 5000
alpha = 1e-5
final_w_spam, final_b_spam, losses = logistic_regression(xTr, yTr, max_iter, alpha)

plt.figure(figsize=(9, 6))
plt.plot(losses)
plt.title("Loss vs. iteration", size=15)
plt.xlabel("Num iteration", size=13)
plt.ylabel("Loss value", size=13)

# evaluate training accuracy
scoresTr = y_pred(xTr, final_w_spam, final_b_spam)
pred_labels = (scoresTr > 0.5).astype(int)
pred_labels[pred_labels != 1] = -1
trainingacc = np.mean(pred_labels == yTr)

# evaluate testing accuracy
scoresTv = y_pred(xTv, final_w_spam, final_b_spam)
pred_labels = (scoresTv > 0.5).astype(int)
pred_labels[pred_labels != 1] = -1
validationacc = np.mean(pred_labels==yTv)
print("Training accuracy %2.2f%%\nValidation accuracy %2.2f%%\n" % (trainingacc*100,validationacc*100))
