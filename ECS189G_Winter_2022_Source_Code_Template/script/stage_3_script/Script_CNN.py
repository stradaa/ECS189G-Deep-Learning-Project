from stages.stage_3_code.Dataset_Loader import Dataset_Loader
from stages.stage_3_code.Method_CNN import Method_CNN
from stages.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from stages.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from stages.stage_3_code.Result_Saver import Result_Saver
from stages.stage_3_code.Setting_TrainTest import TrainTestSplit
from stages.stage_3_code.Setting_TrainTest_ORL import TrainTestSplit_ORL
from stages.stage_3_code.Setting_TrainTest_CIFAR import TrainTestSplit_CIFAR
from stages.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#control = input("chose your dataset. 1 for MNIST, 2 for ORL, and 3 for CIFAR: ")

control = "3"

if control == "1":
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('stage_3_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'

    data_obj.dataset_source_file_name = 'MNIST'

    data = data_obj.load()
    train = data['train']
    test = data['test']


    method_obj = Method_CNN('CNN', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'


    setting_obj = TrainTestSplit('Train Test Split', '')
    # setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    metrics = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Metrics are as follows: ')
    print(' Accuracy, Precision, Recall, F1 ... ' + str(metrics))
    print('************ Finish ************')
    #------------------------------------------------------

if control == "2":
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('stage_3_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'

    data_obj.dataset_source_file_name = 'ORL'

    data = data_obj.load()
    train = data['train']
    test = data['test']

    method_obj = Method_CNN_ORL('CNN', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'


    setting_obj = TrainTestSplit_ORL('Train Test Split', '')
    # setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    metrics = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Metrics are as follows: ')
    print(' Accuracy, Precision, Recall, F1 ... ' + str(metrics))
    print('************ Finish ************')
    #------------------------------------------------------

if control == "3":
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('stage_3_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'

    data_obj.dataset_source_file_name = 'CIFAR'

    data = data_obj.load()
    train = data['train']
    test = data['test']

    method_obj = Method_CNN_CIFAR('CNN', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'


    setting_obj = TrainTestSplit_CIFAR('Train Test Split', '')
    # setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    metrics = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Metrics are as follows: ')
    print(' Accuracy, Precision, Recall, F1 ... ' + str(metrics))
    print('************ Finish ************')
    #------------------------------------------------------