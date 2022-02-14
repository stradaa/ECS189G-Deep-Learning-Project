'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from ..base_class.method import method
from .Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 80
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    #initializing the model architecture
    # Conv2s( channels in, channels out, kernel size, stride)
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.activation_func_1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.activation_func_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(12, 24, 3, padding=1)
        self.activation_func_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_layer_1 = nn.Linear(24*4*4, 330)
        self.activation_func_3 = nn.ReLU()

        self.fc_layer_2 = nn.Linear(330, 250)
        self.activation_func_3 = nn.ReLU()

        self.fc_layer_3 = nn.Linear(250, 100)
        self.activation_func_3 = nn.ReLU()

        self.fc_layer_4 = nn.Linear(100, 50)
        self.activation_func_6 = nn.ReLU()

        self.fc_layer_5 = nn.Linear(50, 10)
        self.activation_func_7 = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''

        # Define how our input travels through the previously defined layers

        h = self.activation_func_1(self.conv1(x))
        #print(h.shape)
        h = self.pool1(h)
        #print(h.shape)
        h = self.activation_func_2(self.conv2(h))
        #print(h.shape)
        h = self.pool2(h)
       # print(h.shape)
        h = self.activation_func_3(self.conv3(h))
        # print(h.shape)
        h = self.pool3(h)
        #print(h.shape)




        #flatten the conv layer
        h = h.view(-1, 24*4*4)

        h = self.activation_func_3(self.fc_layer_1(h))

        h = self.activation_func_3(self.fc_layer_2(h))

        h = self.activation_func_3(self.fc_layer_3(h))

        h = self.activation_func_6(self.fc_layer_4(h))

        # output layer result
        y_pred = self.activation_func_7(self.fc_layer_5(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here
    def train(self, X, y):

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...

            optimizer.zero_grad()

            y_pred = self.forward(torch.FloatTensor(np.array(X)))

            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))

            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()

            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch % 16 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Metrics:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
