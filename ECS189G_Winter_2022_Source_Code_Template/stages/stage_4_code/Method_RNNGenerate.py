import torch.nn as nn
from stages.base_class.method import method
import torch
import time


class RNN(method, nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_words):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_dim = len(vocab_words)
        self.output_dim = len(vocab_words)

        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, text):

        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def binary_accuracy(preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def train(model, iterator, optimizer, criterion, jokes, jokes_size, vocab_words, vocab_index, context, next):
        epoch_loss = 0
        epoch_acc = 0
        X = torch.LongTensor(context)
        y = torch.LongTensor(next)
        sentences = []


        for batch in iterator:
            optimizer.zero_grad()

            predictions = model(X).squeeze(1)

            loss = criterion(predictions, y)

            acc = RNN.binary_accuracy(predictions, y)

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
    def generate(self, input, jokes, jokes_size, vocab_words, vocab_index):
        text = input
        joke = input + []
        for i in range(0, 25):
            encoded = torch.tensor([vocab_words[w] for w in text]).unsqueeze(0)
            predictions = self.forward(encoded).squeeze(0)
            next = vocab_index[torch.argmax(predictions).item()]
            joke.append(next)
            text = text[1:] + [next]
        print("Joke:", joke)
        return joke
