'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import torch

from ..base_class.setting import setting
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class TrainTestSplit(setting):

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        #load training data
        train = loaded_data['train']
        X_train = []
        y_train = []


        for image in train:
            # extract image data and convert
            X_train.append(image['image'])
            # extract label
            y_train.append(image['label'])

        X_train = torch.from_numpy(np.array(X_train))
        X_train = X_train.unsqueeze(1)

        #load lest data
        test = loaded_data['test']
        X_test = []
        y_test = []

        for image in test:
            # extract image data and convert
            X_test.append(image['image'])
            # extract label
            y_test.append(image['label'])

        X_test = torch.from_numpy(np.array(X_test))
        X_test = X_test.unsqueeze(1)

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()
