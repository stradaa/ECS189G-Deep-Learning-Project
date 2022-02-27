from stages.stage_4_code.Dataset_Loader_Generation import Dataset_Loader
from stages.stage_4_code.Method_RNN_Generation import RNN_Generation
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs





data_obj_train = Dataset_Loader('stage_4_data')
data_obj_train.dataset_source_folder_path = r'C:\Users\Sean H\Documents\ecs189' \
                                            r'ECS189G_Winter_2022_Source_Code_Template/' \
                                            r'data\stage_4_data\text_generation'
data_obj_train.dataset_source_file_name = 'data'

train_jokes = data_obj_train.load()
vocab = data_obj_train.unique_words(train_jokes)

INPUT_DIM = len(vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
N_LAYERS = 3

model = RNN_Generation(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS)

optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    state_h, state_c = model.init_state()

    start_time = time.time()

    # perform training and eval

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

