from stages.stage_4_code.Dataset_Loader import Dataset_Loader
from stages.stage_4_code.Method_RNN import RNN
import numpy as np
import torch

if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj_train = Dataset_Loader('stage_4_data')
    data_obj_train.dataset_source_folder_path = r'C:/Users/Alex Estrada/PycharmProjects/ECS189G/' \
                                                r'ECS189G_Winter_2022_Source_Code_Template/' \
                                                r'data/stage_4_data/text_classification/'
    data_obj_train.dataset_source_file_name = ''

    data = data_obj_train.load()

    train_data_pos = data[0]
    train_data_neg = data[1]

    print(train_data_pos.values())
    print(train_data_neg.values())

    # results = data_obj_train.word_embedding(train_data, test_data)
    # print(results[0].vocab.freqs.most_common(20))

    # method_obj = RNN('CNN', '')

    # result_obj = Result_Saver('saver', '')
    # result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    # result_obj.result_destination_file_name = 'classification_result'
    #
    # setting_obj = TrainTestSplit('Train Test Split', '')
    #
    # evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # # ------------------------------------------------------
    #
    # # ---- running section ---------------------------------
    # print('************ Start ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    #
    # metrics = setting_obj.load_run_save_evaluate()
    # print('************ Overall Performance ************')
    # print('RNN Metrics are as follows: ')
    # print(' Accuracy, Precision, Recall, F1 ... ' + str(metrics))
    # print('************ Finish ************')
    # #------------------------------------------------------
