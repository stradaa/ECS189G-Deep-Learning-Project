from stages.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('stage_5_data', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/'

    data_obj.dataset_source_file_name = 'training_stage_2.csv'
    data_obj.testset_source_folder_path = '../../data/stage_5_data/citeseer'
    data_obj.testset_source_file_name = 'testing_stage_2.csv'




    # method_obj = Method_MLP('multi-layer perceptron', '')
    #
    # result_obj = Result_Saver('saver', '')
    # result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    # result_obj.result_destination_file_name = 'prediction_result'
    #
    # setting_obj = TwoFileSplit('Train Test Split', '')
    # # setting_obj = Setting_Tra
    # # in_Test_Split('train test split', '')
    #
    # evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    # print('************ Start ************')
    # setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    # metrics = setting_obj.load_run_save_evaluate()
    # print('************ Overall Performance ************')
    # print('MLP Metrics are as follows: ')
    # print(' Accuracy, Precision, Recall, F1 ... ' + str(metrics))
    # print('************ Finish ************')
    # # ------------------------------------------------------