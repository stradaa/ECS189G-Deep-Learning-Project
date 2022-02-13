
from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    dataset_source_test_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X = []
        y = []
        X2 = []
        y2 = []
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        f2 = open(self.dataset_source_folder_path + self.dataset_source_test_file_name, 'r')

        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split()]
            X.append(elements[1:])
            y.append(elements[0])

        for line in f2:
            line = line.strip('\n')
            elements = [int(i) for i in line.split()]
            X2.append(elements[1:])
            y2.append(elements[0])

        f.close()
        f2.close()
        return {'X': X, 'X_test:': X2, 'y': y, 'y_test': y2}
