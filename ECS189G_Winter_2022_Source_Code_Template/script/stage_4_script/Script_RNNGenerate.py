from stages.stage_4_code.Dataset_LoaderGeneration import Dataset_Loader
from stages.stage_4_code.Method_RNN import RNN
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time

#
# def binary_accuracy(preds, y):
#     """
#     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#     """
#
#     # round predictions to the closest integer
#     rounded_preds = torch.round(torch.sigmoid(preds))
#     correct = (rounded_preds == y).float()  # convert into float for division
#     acc = correct.sum() / len(correct)
#     return acc
#
#
# def train(model, iterator, optimizer, criterion):
#
#     epoch_loss = 0
#     epoch_acc = 0
#
#     # model.train()
#
#     for batch in iterator:
#         optimizer.zero_grad()
#
#         predictions = model(batch.words).squeeze(1)
#
#         loss = criterion(predictions, batch.Label)
#
#         acc = binary_accuracy(predictions, batch.Label)
#
#         loss.backward()
#
#         optimizer.step()
#
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
#
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)
#
#
# def evaluate(model, iterator, criterion):
#
#     epoch_loss = 0
#     epoch_acc = 0
#
#     # model.eval()
#
#     with torch.no_grad():
#         for batch in iterator:
#             predictions = model(batch.words).squeeze(1)
#
#             loss = criterion(predictions, batch.Label)
#
#             acc = binary_accuracy(predictions, batch.Label)
#
#             epoch_loss += loss.item()
#             epoch_acc += acc.item()
#
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)
#
#
# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs
#

if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('stage_4_data')
    data_obj.dataset_source_folder_path = "/Users/Gao_Owen/Documents/GitHub/ECS-189G-Project/ECS189G_Winter_2022_Source_Code_Template/data/stage_4_data/generation_data"
    data_obj.dataset_source_file_name = ''
    jokes, jokes_size, vocab_words, vocab_index, context, next = data_obj.load()

    # INPUT_DIM = len(word_embedding_results[0].vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    # OUTPUT_DIM = 1

    model = RNN(EMBEDDING_DIM, HIDDEN_DIM, vocab_words)

    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # model.train()


    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    # print(f'The model has {count_parameters(model):,} trainable parameters')
    #
    # optimizer = optim.SGD(model.parameters(), lr=1e-3)
    # criterion = nn.BCEWithLogitsLoss()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # criterion = criterion.to(device)

    # N_EPOCHS = 5
    #
    # best_valid_loss = float('inf')
    #
    # # for batch in word_embedding_results[2]:
    # #     optimizer.zero_grad()
    # #
    # #     print(batch.words)
    # #     print(batch.Label)
    #
    # for epoch in range(N_EPOCHS):
    #
    #     start_time = time.time()
    #
    #     train_loss, train_acc = train(model, word_embedding_results[2], optimizer, criterion)
    #     valid_loss, valid_acc = evaluate(model, word_embedding_results[3], criterion)
    #
    #     end_time = time.time()
    #
    #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #
    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(model.state_dict(), 'tut1-model.pt')
    #
    #     print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    #     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    #     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')



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
