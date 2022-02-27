import torch
import torch.nn as nn

class RNN_Generation(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers):

        super(self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(input_size=input_dim,
                           hidden_size=hidden_dim,
                           num_layers=n_layers,
                           dropout=0.2)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x, prev_state):
        embed = self.embedding
        output, state = self.rnn(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

