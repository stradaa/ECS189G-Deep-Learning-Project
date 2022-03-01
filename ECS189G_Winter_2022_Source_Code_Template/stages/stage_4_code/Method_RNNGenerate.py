import torch.nn as nn
from stages.base_class.method import method
import torch
import numpy as np


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_words, batch_size):

        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.seq_size = 3
        self.input_dim = len(vocab_words)
        self.output_dim = len(vocab_words)
        self.dropout = nn.Dropout(0.25)

        # Embedding
        self.embedding = nn.Embedding(self.input_dim, self.hidden_dim)

        # Bi-LSTM
        self.lstm_forward = nn.LSTMCell(hidden_dim, hidden_dim)
        self.lstm_back = nn.LSTMCell(hidden_dim, hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim*2, hidden_dim*2)

        # Fully connected
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)



    def forward(self, text):
        # initialize weights
        hs_forward = torch.zeros(text.size(0), self.hidden_dim)
        cs_forward = torch.zeros(text.size(0), self.hidden_dim)
        hs_back = torch.zeros(text.size(0), self.hidden_dim)
        cs_back = torch.zeros(text.size(0), self.hidden_dim)

        hs_cell = torch.zeros(text.size(0), self.hidden_dim * 2)
        cs_cell = torch.zeros(text.size(0), self.hidden_dim * 2)

        torch.nn.init.kaiming_normal_(hs_forward)
        torch.nn.init.kaiming_normal_(cs_forward)
        torch.nn.init.kaiming_normal_(hs_back)
        torch.nn.init.kaiming_normal_(cs_back)
        torch.nn.init.kaiming_normal_(hs_cell)
        torch.nn.init.kaiming_normal_(cs_cell)

        embedded = self.embedding(text)
        out = embedded.view(self.seq_size, text.size(0), -1)

        forward = []
        back = []

        # forward bilstm
        for i in range(self.seq_size):
            hs_forward, cs_forward = self.lstm_forward(out[i], (hs_forward, cs_forward))
            hs_forward = self.dropout(hs_forward)
            cs_forward = self.dropout(cs_forward)
            forward.append(hs_forward)

        # backward bilstm
        for i in reversed(range(self.seq_size)):
            hs_back, cs_backward = self.lstm_back(out[i], (hs_back, cs_back))
            hs_back = self.dropout(hs_back)
            cs_back = self.dropout(cs_back)
            back.append(hs_back)

        # lstm cell
        for fwd, bwd in zip(forward, back):
            input_tensor = torch.cat((fwd, bwd), 1)
            hs_cell, cs_cell = self.lstm(input_tensor, (hs_cell, cs_cell))

        out = self.fc(hs_cell)
        return out


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def binary_accuracy(model, context, next):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        preds = []
        y = []
        # round predictions to the closest integer
        for i in range(len(context)):
            ct = torch.LongTensor(context[i])
            n = torch.LongTensor(next[i])
            pred = model(ct)
            preds.append(pred)
            y.append(n)
        predsT = torch.LongTensor(preds)
        yT = torch.LongTensor(y)
        correct = (predsT == yT).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def predict(model, v_words, v_index, text, next_words=25):

        words = text.split(' ')

        for i in range(0, next_words):
            x = torch.tensor([[v_index[w] for w in words[i:]]])
            y_pred = model(x)
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(v_words[word_index])

        print(words)

    """
    def train(model, iterator, optimizer, criterion, jokes, jokes_size, vocab_words, vocab_index, context, next):
        epoch_loss = 0
        epoch_acc = 0
        X = torch.LongTensor(context)
        y = torch.LongTensor(next)
        sentences = {}

        for i in range(0, len(X), iterator):
            yield X[i:i + iterator]
            yield y[i:i + iterator]

        print("y has length " + str(len(y)))
        print("x has length " + str(len(X)))

        for batch in sentences:
            optimizer.zero_grad()

            predictions = model(sentences[batch]).forward(1)

            loss = criterion(predictions, batch)

            acc = RNN.binary_accuracy(predictions, batch)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        #return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, model, iterator, criterion):

        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():
            for batch in iterator:
                predictions = model(batch.text).squeeze(1)

                loss = criterion(predictions, batch.label)

                acc = self.binary_accuracy(predictions, batch.label)

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
    """