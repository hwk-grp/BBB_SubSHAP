import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, n_features, n_hidden, dropout):
        super(DNN, self).__init__()

        self.layer1 = nn.Linear(in_features=n_features, out_features=n_hidden)

        # Batch Normalization layer
        self.batch_norm1 = nn.LayerNorm(normalized_shape=n_hidden)

        self.layer2 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self.layer3 = nn.Linear(in_features=n_hidden, out_features=n_hidden)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        self.layer4 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self.layer5 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
        self.layer6 = nn.Linear(in_features=n_hidden, out_features=n_hidden)

        # Last layer - the output dimension is 2 (binary classification)
        self.layer7 = nn.Linear(in_features=n_hidden, out_features=2)

        # Activation function (SeLU)
        self.selu = nn.SELU()

    def forward(self, x):

        x = self.selu(self.layer1(x))
        x = self.batch_norm1(x)

        x = self.selu(self.layer2(x))
        x = self.selu(self.layer3(x))

        x = self.dropout(x)

        x = self.selu(self.layer4(x))
        x = self.selu(self.layer5(x))
        x = self.selu(self.layer6(x))

        x = self.layer7(x)

        return x
