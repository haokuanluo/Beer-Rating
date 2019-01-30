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

NLP = spacy.load('en')
MAX_CHARS = 20000



def pre_tokenize(df):
    df["review/text"] = \
        df['review/text'].str.replace("\n", " ")
    df["review/text"] = \
        df['review/text'].str.replace("\t", " ")
    df["review/text"] = \
        df['review/text'].str.replace(",", " ")
    df["review/text"] = \
        df['review/text'].str.replace(".", " ")
    df["review/text"] = \
        df['review/text'].str.replace("!", " ")
    df["review/text"] = \
        df['review/text'].str.replace("'", " ")
    return df

def prepare_csv(train,test,seed=999,VAL_RATIO=0.2):
    df_train = pd.read_csv(train)
    df_train = pre_tokenize(df_train)

    idx = np.arange(df_train.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx)
    val_size = int(len(idx) * VAL_RATIO)
    df_train.iloc[idx[val_size:], :].to_csv(
        "cache/dataset_train.csv", index=False)
    df_train.iloc[idx[:val_size], :].to_csv(
        "cache/dataset_val.csv", index=False)
    df_test = pd.read_csv(test)
    df_test = pre_tokenize(df_test)
    df_test.to_csv("cache/dataset_test.csv", index=False)











