'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from stages.base_class.dataset import dataset
import torch
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torchtext.legacy import data
import string
import random

# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None):
        super().__init__(dName)

    def load(self):
        null = {"br", "nt", "om", "en", "c"}
        all_data = []
        for x in ['train', 'test']:
            data_dict = {}
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
                data_dict['text'] = text
                data_dict['label'] = y
                all_data.append(data_dict)

        return all_data

    def word_embedding(self, train_data, test_data):
        print("Starting Word Embedding")

        SEED = 1234
        MAX_VOCAB_SIZE = 25_000
        BATCH_SIZE = 64

        train_data = train_data.values()

        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        TEXT = data.Field(tokenize='spacy',
                          tokenizer_language='en_core_web_sm',
                          include_lengths=True)

        LABEL = data.LabelField(dtype=torch.float)

        TEXT.build_vocab(train_data,
                         max_size=MAX_VOCAB_SIZE,
                         vectors="glove.6B.100d",
                         unk_init=torch.Tensor.normal_)

        LABEL.build_vocab(train_data)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            sort_within_batch=True,
            device=device)

        return [TEXT, LABEL, train_iterator, valid_iterator, test_iterator]
