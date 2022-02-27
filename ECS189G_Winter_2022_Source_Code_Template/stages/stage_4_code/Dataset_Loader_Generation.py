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

# https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html


class Dataset_Loader(DS):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None):
        super().__init__(dName)

    def load(self):
        df = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_name)
        df['Joke'] = df.apply(lambda row: word_tokenize(row['Joke']), axis=1)
        return df

    def unique_words(self, dataset):
        df_jokes = dataset['Joke']
        n_words = Counter(df_jokes)
        return sorted(n_words, key=n_words.get, reverse=True)
