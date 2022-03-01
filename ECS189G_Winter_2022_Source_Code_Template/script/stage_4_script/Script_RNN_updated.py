from stages.stage_4_code.Dataset_Loader import Dataset_Loader
from stages.stage_4_code.Method_RNN2 import RNN
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

    train_dict = pickle.load(open('train_pickle', 'rb'))
    test_dict = pickle.load(open('test_pickle', 'rb'))

    word_embedding_results = data_obj_train.word_embedding(train_dict, test_dict)

    INPUT_DIM = len(word_embedding_results[0].vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = word_embedding_results[0].vocab.stoi[word_embedding_results[0].pad_token]

    model = RNN(INPUT_DIM,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT,
                PAD_IDX)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    pretrained_embeddings = word_embedding_results[0].vocab.vectors
    print(pretrained_embeddings.shape)

    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = word_embedding_results[0].vocab.stoi[word_embedding_results[0].unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    print(model.embedding.weight.data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train the model
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 8

    best_valid_loss = float('inf')
    print("START TRAINING")
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = RNN.train(model, word_embedding_results[2], optimizer, criterion)
        valid_loss, valid_acc = RNN.evaluate(model, word_embedding_results[3], criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = RNN.epoch_time(start_time, end_time)

        print("VALID_LOSS:", valid_loss, type(valid_loss))
        print("BEST LOSS:", best_valid_loss, type(best_valid_loss))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    # Results
    model.load_state_dict(torch.load('tut1-model.pt'))
    test_loss, test_acc = RNN.evaluate(model, word_embedding_results[4], criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    RNN.predict_sentiment(model, "This film is terrible")