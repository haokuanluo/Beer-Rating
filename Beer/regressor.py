
import numpy as np

import torch
import torch.nn as nn

from torch.autograd import Variable
from embed import embed_and_pad
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def to_torch(x, dtype, req = False):
  tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x

def load_model(PATH):
    outdim = 5
    args = {
        'emb_size': 300,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.5
    }
    model = SeqLSTMClassify(args,outdim)

    model.load_state_dict(torch.load(PATH), strict=False)
    return model

class SeqLSTMClassify(nn.Module):
    def __init__(self, args, out_dim):
        super(SeqLSTMClassify,self).__init__()

        self.encoder = nn.LSTM(
            input_size=args['emb_size'],
            hidden_size=args['hidden_size'],
            num_layers=args['num_layers'],
            dropout=args['dropout'])

        self.dropout = nn.Dropout(args['dropout'])


        self.fc = nn.Linear(args['hidden_size'], out_dim)
        self.criterion = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)




    def forward(self, inputs, labels):
        # inputs:
        # - ids: seq len x batch, sorted in descending order by length
        #     each row: <S>, first word, ..., last word, </S>
        # - lengths: batch

        embs, lengths = inputs
        # Remove <S> from each sequence
        #embs = self.emb(ids[1:])

        enc_embs_packed = pack_padded_sequence(
            embs, lengths)

        enc_output_packed, enc_state = self.encoder(enc_embs_packed)
        enc_output, lengths = pad_packed_sequence(enc_output_packed)

        # last_enc shape: batch x emb
        last_enc = enc_output[lengths - 1, torch.arange(lengths.shape[0])]
        results = self.fc(self.dropout(last_enc))

        loss = self.criterion(results, labels)



        return enc_state, results, loss

    def learn_once(self,inputs,labels): # inputs = embs,lengths
        self.opt.zero_grad()
        enc_state, results, loss = self.forward(inputs, labels)
        loss.backward()
        self.opt.step()
        return loss

    def predict(self,inputs,labels,perm_idx):
        enc_state, results, loss = self.forward(inputs, labels)
        _, unperm_idx = perm_idx.sort(0)
        results = results[unperm_idx]
        return results.data.cpu().numpy()




def train(df,EPOCH=1):
    outdim = 5
    args = {
        'emb_size': 300,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.5
    }
    BATCH_SIZE = 64
    model = SeqLSTMClassify(args,outdim)

    for epoch in range(EPOCH):
        idx = np.arange(len(df.index))
        np.random.shuffle(idx)

        for i in range(0, len(df.index), BATCH_SIZE):
            print(i)
            seq_tensor, y, seq_lengths, perm_idx = embed_and_pad(df, idx[i:i + BATCH_SIZE])
            loss = model.learn_once((seq_tensor, seq_lengths), y)
            print(loss.data.cpu().numpy())


    return model


def eval(df,model):
    seq_tensor, y, seq_lengths, perm_idx = embed_and_pad(df, np.arange(len(df.index)))
    loss = model.learn_once((seq_tensor, seq_lengths), y)
    print(loss.data.cpu().numpy())
    return loss.data.cpu().numpy()

def predict(df,model):
    seq_tensor, y, seq_lengths, perm_idx = embed_and_pad(df, np.arange(len(df.index)))
    return model.predict((seq_tensor, seq_lengths), y, perm_idx)