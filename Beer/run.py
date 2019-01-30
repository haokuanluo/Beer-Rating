# coding: utf-8
import pandas as pd

import torch

from regressor import train, load_model
from regressor import predict


def get_predictions(model):
    ans = []
    df = pd.read_csv('cache/dataset_test.csv')
    res = predict(df,model)
    for i in range(len(df.index)):
        ro = [df.loc[i,'index']]
        ro.extend(res[i,:])
        ans.append(ro)
    return ans

def write_csv():
    model = load_model('trained_model')
    ans = get_predictions(model)
    f = open('submission.csv', 'w')
    f.write('index,review/appearance,review/aroma,review/overall,review/palate,review/taste\n')  # Give your csv text here.
    for rows in ans:
        line = ",".join([str(x) for x in rows])+'\n'
        f.write(line)


    f.close()


def start_train():
    filename = 'cache/dataset_train.csv'
    df = pd.read_csv(filename)
    df = df[pd.notnull(df['review/text'])].reset_index()
    model = train(df, EPOCH=1)
    PATH = 'trained_model'
    torch.save(model.state_dict(), PATH)

if __name__ == '__main__':
    write_csv()


