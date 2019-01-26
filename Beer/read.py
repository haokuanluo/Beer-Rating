# coding: utf-8
import pandas as pd

import torch
from torchtext import data
from torchtext import datasets
import random
import re
import spacy
import numpy as np


def get_data(filename):
    df = pd.read_csv(filename,low_memory= False)
    df["review/text"] = \
        df['review/text'].str.replace("\n", " ")
    df["review/text"] = \
        df['review/text'].str.replace("\t", " ")
    return df

def tokenizer(comment):
    NLP = spacy.load('en')
    MAX_CHARS = 20000
    comment = re.sub(
        r"[\*\"\n\\…\+\-\/\=\(\)‘:\[\]\|’\!;]", " ",
        str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return [
        x.text for x in NLP.tokenizer(comment) if x.text != " "]

def prepare_csv(train,test,seed=999,VAL_RATIO=0.2):
    df_train = pd.read_csv(train)
    df_train["review/text"] = \
        df_train['review/text'].str.replace("\n", " ")
    df_train["review/text"] = \
        df_train['review/text'].str.replace("\t", " ")
    idx = np.arange(df_train.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx)
    val_size = int(len(idx) * VAL_RATIO)
    df_train.iloc[idx[val_size:], :].to_csv(
        "cache/dataset_train.csv", index=False)
    df_train.iloc[idx[:val_size], :].to_csv(
        "cache/dataset_val.csv", index=False)
    df_test = pd.read_csv(test)
    df_test["review/text"] = \
        df_test['review/text'].str.replace("\n", " ")
    df_test["review/text"] = \
        df_test['review/text'].str.replace("\t", " ")
    df_test.to_csv("cache/dataset_test.csv", index=False)

def get_dataset(train,test,fix_length=100, lower=False, vectors=None):
    if vectors is not None:
        # pretrain vectors only supports all lower cases
        lower = True
    print('preparing')
    prepare_csv(train,test)
    print('done')
    comment = data.Field(
        sequential=True,
        fix_length=fix_length,
        tokenize=tokenizer,
        pad_first=True,
        dtype=torch.cuda.LongTensor,
        lower=lower
    )
    print('tokening')
    if torch.cuda.is_available():
        mdtype = torch.cuda.HalfTensor
    else:
        mdtype = torch.HalfTensor
    train, val = data.TabularDataset.splits(
        path='cache/', format='csv', skip_header=True,
        train='dataset_train.csv', validation='dataset_val.csv',
        fields=[
            ('index', None),
            ('review/text', comment),
            ('review/appearance', data.Field(
                use_vocab=False, sequential=False,
                dtype=mdtype)),
            ('review/aroma', data.Field(
                use_vocab=False, sequential=False,
                dtype=mdtype)),
            ('review/overall', data.Field(
                use_vocab=False, sequential=False,
                dtype=mdtype)),
            ('review/palate', data.Field(
                use_vocab=False, sequential=False,
                dtype=mdtype)),
            ('review/taste', data.Field(
                use_vocab=False, sequential=False,
                dtype=mdtype)),
        ])
    print("Reading test csv file...")
    test = data.TabularDataset(
        path='cache/dataset_test.csv', format='csv',
        skip_header=True,
        fields=[
            ('id', None),
            ('comment_text', comment)
        ])
    print("Building vocabulary...")
    comment.build_vocab(
        train, val, test,
        max_size=20000,
        min_freq=50,
        vectors=vectors
    )
    print("Done preparing the datasets")
    return train, val, test