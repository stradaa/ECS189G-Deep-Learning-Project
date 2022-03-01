import torch
import torch.nn as nn

class RNN_Generation(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, vocab_size):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            n_layers)
        self.fc = nn.Linear(n_layers, vocab_size)
        pass

