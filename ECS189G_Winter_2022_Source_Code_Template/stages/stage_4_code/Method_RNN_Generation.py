import torch
import torch.nn as nn

class RNN_Generation(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):

        super(self).__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(self.embedding_dim)
