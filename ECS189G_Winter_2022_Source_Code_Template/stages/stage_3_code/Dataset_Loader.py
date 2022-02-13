'''
Concrete IO class for a specific dataset
'''

import pickle
from ..base_class.dataset import dataset
import matplotlib.pyplot as plt


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__((dName, dDescription))

    def load(self):
        print('loading data...')
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)

        print('data loaded.')
        return {'test': data['test'], 'train': data['train']}