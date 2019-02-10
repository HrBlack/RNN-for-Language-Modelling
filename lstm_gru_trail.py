#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:41:31 2019

@author: s1583620
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:31:56 2019

@author: s1583620
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from rnnmath import *
from sys import stdout
torch.manual_seed(1)

data_folder = '/afs/inf.ed.ac.uk/user/s15/s1583620/nlu-coursework/data'
 
train_size = 25000
dev_size = 1000
vocab_size = 2000
hdim = 50
Max_epochs = 30

vocab = pd.read_table(data_folder + "/vocab.wiki.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
num_to_word = dict(enumerate(vocab.index[:vocab_size]))     #Mapping from index to word
word_to_num = invert_dict(num_to_word)      # Mapping from word to index

docs = load_lm_dataset(data_folder + '/wiki-train.txt')
S_train = docs_to_indices(docs, word_to_num, 1, 1)
X_train, D_train = seqs_to_lmXY(S_train)

docs = load_lm_dataset(data_folder + '/wiki-dev.txt')
S_dev = docs_to_indices(docs, word_to_num, 1, 1)
X_dev, D_dev = seqs_to_lmXY(S_dev)

X_train = X_train[:train_size]
D_train = D_train[:train_size]      # Targets, i.e. one-offset X_train
X_dev = X_dev[:dev_size]
D_dev = D_dev[:dev_size]






def batchify(data,batch_size):
    """
    Batchify the data, input should come from X_Train[:batch_size], i.e. a list
    of size batch_size. In the list are arrays.
    """
    n_batch = len(data)//batch_size   # Number of batches
    data = data[:n_batch*batch_size]    # Discard the extra data
    #====Till now, data is still a list
    tensor_data = torch.zeros()
    for i in range(len(data)):
        pass
    

def data_to_tensor(seq,batch_size = 1):
    """
    Input should be one sentence, e.g. X_train[0]
    Output will be an variable tensor
    """
    tensor = torch.LongTensor(seq)
    return autograd.Variable(tensor)




class RNNTagger(nn.Module): 
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(RNNTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.RNN(embedding_dim, hidden_dim)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
    
    def one_hot(self,sentence):
        """
            One-hot embedding, now can only receive one sentence
        """
        sentence = sentence.reshape((len(sentence),1))
        one_hot = torch.zeros(len(sentence),self.vocab_size).scatter_(dim=1,index=sentence,value=1)
        return one_hot
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
        
    def forward(self, sentence):
        #embeds = self.word_embeddings(sentence)
        embeds = self.one_hot(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores



class LSTMTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
    
    def one_hot(self,sentence):
        """
            One-hot embedding, now can only receive one sentence
        """
        sentence = sentence.reshape((len(sentence),1))
        one_hot = torch.zeros(len(sentence),self.vocab_size).scatter_(dim=1,index=sentence,value=1)
        return one_hot
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
        
    def forward(self, sentence):
        #embeds = self.word_embeddings(sentence)
        embeds = self.one_hot(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores


class GRUTagger(nn.Module): 
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(GRUTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.GRU(embedding_dim, hidden_dim)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
    
    def one_hot(self,sentence):
        """
            One-hot embedding, now can only receive one sentence
        """
        sentence = sentence.reshape((len(sentence),1))
        one_hot = torch.zeros(len(sentence),self.vocab_size).scatter_(dim=1,index=sentence,value=1)
        return one_hot
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
        
    def forward(self, sentence):
        #embeds = self.word_embeddings(sentence)
        embeds = self.one_hot(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

f = open('result.txt','w')


print('===================RNN Model:=========================')
f.write('===================RNN Model:=========================\n')

model_rnn = RNNTagger(vocab_size, hdim, vocab_size, vocab_size)
loss_function_rnn = nn.NLLLoss()
optimizer_rnn = optim.SGD(model_rnn.parameters(), lr=0.2)


loss_table_rnn = []
loss_dev_rnn = []

t0 = time.time()
for epoch in range(Max_epochs):
    loss_sum = 0
    loss_dev = 0
    for idx in range(train_size):
        X_tensor = data_to_tensor(X_train[idx])
        Y_tensor = data_to_tensor(D_train[idx])
        model_rnn.zero_grad()
        model_rnn.hidden = model_rnn.init_hidden()
        
        pred_scores = model_rnn(X_tensor)
        loss = loss_function_rnn(pred_scores,Y_tensor)
        loss.backward()
        optimizer_rnn.step()
        
        loss_sum += np.array(loss.data)
    loss_table_rnn.append(loss_sum/train_size)  
        
    if epoch%2==0:
        for idx in range(dev_size):
            X_tensor = data_to_tensor(X_dev[idx])
            Y_tensor = data_to_tensor(D_dev[idx])
            pred_dev = model_rnn(X_tensor)
            loss_d = loss_function_rnn(pred_dev,Y_tensor)
            loss_dev += np.array(loss_d.data)  
        loss_dev_rnn.append(loss_dev/dev_size)  
        print('epoch= ',epoch,'loss is: ',loss_sum/train_size,'dev: ',loss_dev/dev_size)
        f.write('epoch= %s'%epoch)
        tmp = loss_sum/train_size
        f.write('loss is: %s\t'% tmp)
        tmp = loss_dev/dev_size
        f.write('dev loss is: %s\n'%tmp)
t1 = time.time()
t_rnn = t1-t0
print('Total time: ',t1-t0)
f.write('Total time: %s\n'%t_rnn)
    

print('===================LSTM Model:=========================')
f.write('===================LSTM Model:=========================\n')
model = LSTMTagger(vocab_size, hdim, vocab_size, vocab_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.2)

loss_table = []
loss_dev_table = []
t0 = time.time()
for epoch in range(Max_epochs):
    loss_sum = 0
    loss_dev = 0
    for idx in range(train_size):
        X_tensor = data_to_tensor(X_train[idx])
        Y_tensor = data_to_tensor(D_train[idx])
        model.zero_grad()
        model.hidden = model.init_hidden()
        
        pred_scores = model(X_tensor)
        loss = loss_function(pred_scores,Y_tensor)
        loss.backward()
        optimizer.step()
        
        loss_sum += np.array(loss.data)
    loss_table.append(loss_sum/train_size)

    if epoch%2==0:
        for idx in range(dev_size):
            X_tensor = data_to_tensor(X_dev[idx])
            Y_tensor = data_to_tensor(D_dev[idx])
            pred_dev = model(X_tensor)
            loss_d = loss_function(pred_dev,Y_tensor)
            loss_dev += np.array(loss_d.data)
        loss_dev_table.append(loss_dev/dev_size)
        print('epoch= ',epoch,'loss is: ',loss_sum/train_size,'dev: ',loss_dev/dev_size)
        f.write('epoch= %s'%epoch)
        tmp = loss_sum/train_size
        f.write('loss is: %s\t'% tmp)
        tmp = loss_dev/dev_size
        f.write('dev loss is: %s\n'%tmp)

t1 = time.time()
t_lstm = t1-t0
print('Total time: ',t1-t0)
f.write('Total time: %s\n'%t_lstm)





print('===================GRU Model:=========================')
f.write('===================GRU Model:=========================\n')
model_gru = GRUTagger(vocab_size, hdim, vocab_size, vocab_size)
loss_function_gru = nn.NLLLoss()
optimizer_gru = optim.SGD(model_gru.parameters(), lr=0.2)

loss_table_gru = []
loss_dev_gru=[]
t0 = time.time()
for epoch in range(Max_epochs):
    loss_sum = 0
    loss_dev = 0
    for idx in range(train_size):
        X_tensor = data_to_tensor(X_train[idx])
        Y_tensor = data_to_tensor(D_train[idx])
        model_gru.zero_grad()
        model_gru.hidden = model_gru.init_hidden()
        
        pred_scores = model_gru(X_tensor)
        loss = loss_function_gru(pred_scores,Y_tensor)
        loss.backward()
        optimizer_gru.step()
        
        loss_sum += np.array(loss.data)
    loss_table_gru.append(loss_sum/train_size)

    if epoch%2==0:
        for idx in range(dev_size):
            X_tensor = data_to_tensor(X_dev[idx])
            Y_tensor = data_to_tensor(D_dev[idx])
            pred_dev = model_gru(X_tensor)
            loss_d = loss_function_gru(pred_dev,Y_tensor)
            loss_dev += np.array(loss_d.data)   
        loss_dev_gru.append(loss_dev/dev_size)
        print('epoch= ',epoch,'loss is: ',loss_sum/train_size,'dev: ',loss_dev/dev_size)
        f.write('epoch= %s'%epoch)
        tmp = loss_sum/train_size
        f.write('loss is: %s\t'% tmp)
        tmp = loss_dev/dev_size
        f.write('dev loss is: %s\n'%tmp)

t1 = time.time()
t_gru = t1-t0
print('Total time: ',t1-t0)
f.write('Total time: %s\n'%t_gru)

f.close()


'''
x_range = np.arange(0,15)*2
fig1 = plt.figure(figsize=(8, 3))
plt.subplot(1,2,1)
plt.title('Training Loss of RNN, LSTM, and GRU')
plt.plot(loss_table_rnn,'g+-',label='RNN')
plt.plot(loss_table,'rx-',label='LSTM')
plt.plot(loss_table_gru,'b.-',label='GRU')
plt.legend()
plt.xlabel('No. of epochs.')
plt.ylabel('Loss value')
plt.grid(True)

plt.subplot(1,2,2)
plt.title('Dev Loss of RNN, LSTM, and GRU')
plt.plot(x_range,loss_dev_rnn,'g+-',label='RNN')
plt.plot(x_range,loss_dev_table,'rx-',label='LSTM')
plt.plot(x_range,loss_dev_gru,'b.-',label='GRU')
plt.legend()
plt.xlabel('No. of epochs.')
plt.ylabel('Loss value')
plt.grid(True)

fig1.tight_layout()
fig1.savefig('rnn_lstm_gru.pdf')
'''















