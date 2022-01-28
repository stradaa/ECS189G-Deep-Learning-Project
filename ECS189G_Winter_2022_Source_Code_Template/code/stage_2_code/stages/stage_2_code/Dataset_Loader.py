'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from ..base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    dataset_source_file_test_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X = []
        y = []
        X_test = []
        y_test = []
        f1 = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        f2 = open(self.dataset_source_folder_path + self.dataset_source_file_test_name, 'r')
        for line in f1:
            line = line.strip('\n')

            elements = [int(i) for i in line.split()]

            X.append(elements[1:])
            y.append(elements[0])

        for line in f2:
            line = line.strip('\n')

            elements = [int(i) for i in line.split()]

            X_test.append(elements[1:])
            y_test.append(elements[0])

        f1.close()
        f2.close()
        return {'X': X, 'X_test': X_test, 'y': y, 'y_test': y_test}