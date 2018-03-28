import numpy as np
from scipy.sparse import rand as sprand
from scipy.sparse import lil_matrix
import torch
from torch.autograd import Variable
import pandas as pd
from math import sqrt

#use_gpu = torch.cuda.is_available()

def get_movielens_ratings(df):
    n_users = max(df.user_id.unique())
    n_items = max(df.item_id.unique())

    interactions = lil_matrix( (n_users,n_items), dtype=float) #np.zeros((n_users, n_items))
    for row in df.itertuples():
        interactions[row[1] - 1, row[2] - 1] = row[3]
    return interactions


names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = []
test_ratings = []
for i in range(5):
    df_train = pd.read_csv('ml-10M100K/r'+str(i+1)+'.train', sep='::', names=names,engine='python')
    df_test = pd.read_csv('ml-10M100K/r'+str(i+1)+'.test', sep='::', names=names,engine='python')
    ratings.append(get_movielens_ratings(df_train))
    test_ratings.append(get_movielens_ratings(df_test))

class MatrixFactorization(torch.nn.Module):

    def __init__(self, n_users, n_items, n_factors=5):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users,
                                               n_factors,
                                               sparse=False)
        self.item_factors = torch.nn.Embedding(n_items,
                                               n_factors,
                                               sparse=False)
        # Also should consider fitting overall bias (self.mu term) and both user and item bias vectors
        # Mu is 1x1, user_bias is 1xn_users. item_bias is 1xn_items

    # For convenience when we want to predict a sinble user-item pair.
    def predict(self, user, item):
        # Need to fit bias factors
        return (pred + self.user_factors(user) * self.item_factors(item)).sum(1)

    # Much more efficient batch operator. This should be used for training purposes
    def forward(self, users, items):
        # Need to fit bias factors
        return torch.mm(self.user_factors(users),torch.transpose(self.item_factors(items),0,1))

class BiasedMatrixFactorization(torch.nn.Module):

    def __init__(self, n_users, n_items, n_factors=5):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users,
                                               n_factors,
                                               sparse=False)
        self.item_factors = torch.nn.Embedding(n_items,
                                               n_factors,
                                               sparse=False)
        self.item_biases = torch.nn.Embedding(n_items,
                                              1,
                                              sparse=False)



    def predict(self, users, items):
        pred = self.user_factors(users)*self.item_factors(items).sum(1)
        pred += self.item_biases(items)
        return pred
    def forward(self, users, items):
        return torch.mm(self.user_factors(users),torch.transpose(self.item_factors(items),0,1)) + self.item_biases(items)


def get_batch(batch_size,ratings):
    # Sort our data and scramble it
    rows, cols = ratings.shape
    p = np.random.permutation(rows)

    # create batches
    sindex = 0
    eindex = batch_size
    while eindex < rows:
        batch = p[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= rows:
        batch = p[sindex:rows]
        yield batch

EPOCH = 30
BATCH_SIZE = 1000 #50

loss_func = torch.nn.MSELoss()

def run_train_epoch(ratings):
    count = 0
    total_loss = 0
    for i,batch in enumerate(get_batch(BATCH_SIZE, ratings)):

        count += 1
        # Set gradients to zero
        reg_loss_func.zero_grad()

        # Turn data into variables
        interactions = Variable(torch.FloatTensor(ratings[batch, :].toarray()))
        rows = Variable(torch.LongTensor(batch))
        cols = Variable(torch.LongTensor(np.arange(ratings.shape[1])))

        # Predict and calculate loss
        predictions = model(rows, cols)
        loss = loss_func(predictions, interactions)
        total_loss += loss
        # Backpropagate
        loss.backward()

        # Update the parameters
        reg_loss_func.step()
    print('train avg loss is %f'%(total_loss/count))

def run_test_epoch(ratings):
    for i,batch in enumerate(get_batch(BATCH_SIZE, ratings)):
        # Turn data into variables
        interactions = Variable(torch.FloatTensor(ratings[batch, :].toarray()))
        rows = Variable(torch.LongTensor(batch))
        cols = Variable(torch.LongTensor(np.arange(ratings.shape[1])))

        # Predict and calculate loss
        predictions = model(rows, cols)
        loss = loss_func(predictions, interactions)
        # Backpropagate
        #loss.backward()

        # Update the parameters
        #reg_loss_func.step()
    return sqrt(loss)

weight = [0.001, 0.01, 0.1]
for i in range(4):
    print('for train set %d'%(i+1)+':')
    print('*******************************')
    model = MatrixFactorization(ratings[i].shape[0], ratings[i].shape[1], n_factors=5)
    #model = BiasedMatrixFactorization(ratings[i].shape[0], ratings[i].shape[1], n_factors=5)
    for w in range(3):
        reg_loss_func = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay = weight[w])
        for j in range(EPOCH):
            print('with λ %.3f'%weight[w] + ' EPOCH %d'%j+":")
            run_train_epoch(ratings[i])

        print('with test set %i'%(i+1))
        loss = run_test_epoch(test_ratings[i])
        print('rmse loss is: %f'%loss+' with λ=%.3f'%weight[w]+' train data %d'%(i+1))
        print('---------------------------')
