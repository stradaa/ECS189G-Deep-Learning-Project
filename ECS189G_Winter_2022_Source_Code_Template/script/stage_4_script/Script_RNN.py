from stages.stage_4_code.Dataset_Loader import Dataset_Loader
from stages.stage_4_code.Method_RNN import RNN
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time




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

    # data = data_obj_train.load2()

    # #
    # train_dict = data[0]
    # test_dict = data[1]
    #
    # pickle_out = open("train_pickle", 'wb')
    # pickle_out2 = open("test_pickle", 'wb')
    # pickle.dump(train_dict, pickle_out)
    # pickle.dump(test_dict, pickle_out2)
    # pickle_out.close()
    # pickle_out2.close()

    train_dict = pickle.load(open('train_pickle', 'rb'))
    test_dict = pickle.load(open('test_pickle', 'rb'))

    word_embedding_results = data_obj_train.word_embedding(train_dict, test_dict)

    INPUT_DIM = len(word_embedding_results[0].vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.SGD(model.parameters(), lr=0.75)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 80

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = RNN.train(model, word_embedding_results[2], optimizer, criterion)
        valid_loss, valid_acc = RNN.evaluate(model, word_embedding_results[3], criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = RNN.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load('tut2-model.pt'))

    test_loss, test_acc = RNN.evaluate(model, word_embedding_results[4], criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
