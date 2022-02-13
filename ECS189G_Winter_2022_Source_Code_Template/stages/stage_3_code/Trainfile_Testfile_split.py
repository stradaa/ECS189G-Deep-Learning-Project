'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from ..base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np


class TwoFileSplit(setting):
    fold = 3

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        X_train = loaded_data['X']
        X_test = loaded_data['testX']
        y_train = loaded_data['y']
        y_test  = loaded_data['testY']

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate()
