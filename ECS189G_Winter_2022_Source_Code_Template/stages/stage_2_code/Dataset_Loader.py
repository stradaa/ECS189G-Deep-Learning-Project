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
    testset_source_folder_path = None
    testset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading training data...')
        X = []
        y = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')

        for line in f:
            line = line.strip('\n')


            elements = [int(i) for i in line.split(',')]

            elements = [int(i) for i in line.split(",")]


            X.append(elements[1:])
            y.append(elements[0])
        f.close()

        print('loading testing data...')
        testX = []
        testY = []
        f2 = open(self.testset_source_folder_path + self.testset_source_file_name, 'r')

        for line in f2:

            elements = [int(i) for i in line.split(',')]

            line = line.strip('\n')

            elements = [int(i) for i in line.split(",")]


            testX.append(elements[1:])
            testY.append(elements[0])

        f2.close()

        return {'X': X, 'y': y, 'testX': testX, 'testY': testY}


