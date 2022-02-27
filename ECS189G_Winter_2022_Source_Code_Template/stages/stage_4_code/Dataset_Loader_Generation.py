'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from stages.base_class.dataset import dataset as DS
from script.stage_4_script.DataFrameDataset import DataFrameDataset
import torch
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torchtext.legacy import data
from torchtext.legacy.data import Field, Dataset, Example
import pandas as pd
import numpy as np
import string
import random
from collections import Counter

# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb


class Dataset_Loader(DS):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None):
        super().__init__(dName)

    def load(self):
        df = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_name)
        """
        ID = []
        Joke = []
        
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        next(f)
        
        for line in f:
            elements = line.split(',', 1)
            ID.append(int(elements[0]))
            no_quotes = elements[1][1:-1]
            for sent in no_quotes:
                tokens = word_tokenize(sent)
                Joke.append(tokens)
        f.close()
        
        return {'ID': ID, 'Joke': Joke}
        """
        df['Joke'] = df.apply(lambda row: word_tokenize(row['Joke']), axis=1)
        return df

    def unique_words(self, dataset):
        df_jokes = dataset['Joke']
        TEXT = data.Field(tokenize='spacy',
                          tokenizer_language='en_core_web_sm')
        TEXT.build_vocab(df_jokes, max_size=25000, vectors="glove.6B.100d")