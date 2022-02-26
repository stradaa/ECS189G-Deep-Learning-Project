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
        # Labels
        array_int = np.ones(len(train_data['pos']), dtype=int)
        array_int2 = np.zeros(len(train_data['neg']), dtype=int)

        array_int3 = np.ones(len(test_data['pos']), dtype=int)
        array_int4 = np.zeros(len(test_data['neg']), dtype=int)

        # Pos and neg training
        df1 = pd.DataFrame(train_data['pos'], columns=['words'])
        df11 = pd.DataFrame(array_int, columns=['Label'])
        df2 = pd.DataFrame(train_data['neg'], columns=['words'])
        df22 = pd.DataFrame(array_int2, columns=['Label'])
        # Pos and neg testing
        df3 = pd.DataFrame(test_data['pos'], columns=['words'])
        df33 = pd.DataFrame(array_int3, columns=['Label'])
        df4 = pd.DataFrame(test_data['neg'], columns=['words'])
        df44 = pd.DataFrame(array_int4, columns=['Label'])

        # concat
        df_train_data_pos = pd.concat([df1, df11], axis=1)
        df_train_data_neg = pd.concat([df2, df22], axis=1)
        df_test_data_pos = pd.concat([df3, df33], axis=1)
        df_test_data_neg = pd.concat([df4, df44], axis=1)

        df_train = pd.concat([df_train_data_pos, df_train_data_neg], ignore_index=True)
        df_test = pd.concat([df_test_data_pos, df_test_data_neg], ignore_index=False)

        # print(df_train.head())
        SEED = 1234

        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        TEXT = data.Field(tokenize='spacy',
                          tokenizer_language='en_core_web_sm')
        TEXT.build_vocab(df_train, max_size=25000, vectors="glove.6B.100d")
        LABEL = data.LabelField(dtype=torch.float)
        LABEL.build_vocab(df_train)
        fields = {'Label': LABEL, 'words': TEXT}

        train_ds = DataFrameDataset(df_train, fields)
        test_ds = DataFrameDataset(df_test, fields)

        train_data, valid_data = train_ds.split(random_state=random.seed(SEED))

        BATCH_SIZE = 64

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_ds),
            batch_size=BATCH_SIZE,
            device=device)

        return TEXT, LABEL, train_iterator, valid_iterator, test_iterator

