# coding: utf-8
import torch
from torchtext import data
from torchtext import datasets
import random
import re
import numpy as np
import spacy
from gensim.models import KeyedVectors
from torch.autograd import Variable
#from regressor import to_torch
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
#from __future__ import unicode_literals

def to_torch(x, dtype='float', req = False):
  tor_type = torch.LongTensor if dtype == "int" else torch.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x


embed_dim = 300
def embed(words):
    if(type(words)!=str):

        list = []
        list.append(np.random.random_sample((300,)))
        return list



    words = str.split(words)
    list = []
    for word in words:
        #print('*'+word+'*')
        try:
            list.append(model[word])
            #list.append(np.random.random_sample((300,)))
            #print(model[word])
        except:
            pass
    return list

def embed_and_pad(df,idx):
    x=[]
    y=df.loc[idx,['review/appearance','review/aroma','review/overall','review/palate','review/taste']].values
    maxlen = 0
    for i in idx:
        embedded = embed(df.loc[i,'review/text'])
        maxlen = max(maxlen,len(embedded))
        x.append(embedded)
    seq_tensor = torch.zeros((len(idx), maxlen, embed_dim)).float()
    for i in range(len(idx)):
        seq_tensor[i,:len(x[i]),:]=to_torch(np.array(x[i]))
    seq_lengths = torch.LongTensor([len(seq) for seq in x])
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    #print(y)
    y = y[perm_idx,:]
    seq_tensor = seq_tensor.transpose(0,1)
    y = to_torch(y)
    return seq_tensor, y, seq_lengths, perm_idx






