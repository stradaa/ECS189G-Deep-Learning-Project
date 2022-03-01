'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from stages.base_class.dataset import dataset as DS
# from script.stage_4_script.DataFrameDataset import DataFrameDataset
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
import itertools
import collections


# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb


class Dataset_Loader(DS):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    vocab_words = {}
    vocab_index = {}


    def __init__(self, dName=None):
        super().__init__(dName)

    def load(self):
        path = self.dataset_source_folder_path
        jokes = []
        jokes_size = []
        with open(path) as file:
            for i,buff in enumerate(file):
                joke = buff.split(',',1)[1].replace("\n", "").lower().split()
                jokes.append(joke)
                jokes_size.append(len(joke))
        # print(jokes)
        words = list(itertools.chain.from_iterable(jokes))
        word_count = collections.Counter(words).most_common()
        # print(word_count)
        # word_count = Counter(words).most_common()
        for i, word in enumerate(word_count):
            self.vocab_words[word[0]] = i
            self.vocab_index[i] = word[0]
        context = []
        next = []

        for joke in jokes:
            encoded_joke = [self.vocab_words[word] for word in joke]
            for i in range(len(encoded_joke) - 3):
                context.append(encoded_joke[i:i+3])
                next.append(encoded_joke[i+3])
        # print("jokes:", jokes)
        # print("[[self.vocab_index[ind] for ind in joke] for joke in context]: ", [[self.vocab_index[ind] for ind in joke] for joke in context])
        # print("next:", next)
        # print([self.vocab_index[ind] for ind in next])
        # print("context", context)
        # print("next", next)

        return jokes, jokes_size, self.vocab_words, self.vocab_index, context, next

