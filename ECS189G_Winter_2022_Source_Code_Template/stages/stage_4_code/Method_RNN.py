import torch.nn as nn
import torch


class RNN(nn.Module):

    INPUT_DIM = None
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1


    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))

    def binary_accuracy(preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def train(model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for batch in iterator:
            optimizer.zero_grad()

            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = RNN.binary_accuracy(predictions, batch.label)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(model, iterator, criterion):

        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():
            for batch in iterator:
                predictions = model(batch.text).squeeze(1)

                loss = criterion(predictions, batch.label)

                acc = binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)