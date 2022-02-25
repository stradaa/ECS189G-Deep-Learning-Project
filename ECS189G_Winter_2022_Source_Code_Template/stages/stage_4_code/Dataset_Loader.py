'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from stages.base_class.dataset import dataset as DS
import torch
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torchtext.legacy import data
from torchtext import datasets
import string
import random

# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb


class Dataset_Loader(DS):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None):
        super().__init__(dName)

    def load(self):
        null = {"br", "nt", "om", "en", "c"}
        train_dict = {}
        test_dict = {}
        text_data = []
        for x in ['train', 'test']:
            for y in ['/pos', '/neg']:
                updated_path = self.dataset_source_folder_path + x + y
                text_files = os.listdir(updated_path)
                print('loading data in...', updated_path)

                text = []
                for i in text_files:
                    temp = open(updated_path + "/" + i, encoding="utf8")
                    file = temp.read()
                    temp.close()

                    # split into words
                    tokens = word_tokenize(file)
                    # convert to lower case
                    tokens = [w.lower() for w in tokens]
                    # remove punctuation
                    table = str.maketrans('', '', string.punctuation)
                    stripped = [w.translate(table) for w in tokens]
                    # remove tokens not alphabetic
                    words = [word for word in stripped if word.isalpha()]
                    # filtering stop words
                    stop_words = set(stopwords.words('english'))
                    words = [w for w in words if not w in stop_words]
                    # further filter in those examined
                    words = [w for w in words if not w in null]

                    text.extend(words)
                text_data.append(text)

            if x == 'train':
                train_dict['pos'] = text_data[0]
                train_dict['neg'] = text_data[1]
            else:
                test_dict['pos'] = text_data[2]
                test_dict['neg'] = text_data[3]

        return train_dict, test_dict


    def word_embedding(self, train_data, test_data):
        print("Starting Word Embedding")

        SEED = 1234
        MAX_VOCAB_SIZE = 25_000
        BATCH_SIZE = 64

        TEXT = data.Field(tokenize='spacy',
                          tokenizer_language='en_core_web_sm',
                          include_lengths=True)

        LABEL = data.LabelField(dtype=torch.float)

        # Load the dataset

        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            sort_within_batch=True,
            device=device)

        return [TEXT, LABEL, train_iterator, valid_iterator, test_iterator]
