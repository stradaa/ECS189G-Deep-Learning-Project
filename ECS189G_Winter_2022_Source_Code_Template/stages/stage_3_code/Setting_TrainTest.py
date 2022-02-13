'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from ..base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np


class TrainTestSplit(setting):

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        train = loaded_data['train']
        X_train = []
        y_train = []

        for image in train:
            # extract image data
            X_train.append(image['image'])
            # extract label
            y_train.append(image['label'])


        test = loaded_data['test']
        X_test = []
        y_test = []

        for image in test:
            # extract image data
            X_train.append(image['image'])
            # extract label
            y_train.append(image['label'])


        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()
