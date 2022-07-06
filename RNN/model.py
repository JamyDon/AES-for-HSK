import math

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.dropout_rate = config.dropout_rate
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_size)

        self.rnn = nn.RNN(input_size=config.embedding_size,
                           hidden_size=config.hidden_size,
                           batch_first=True,
                           num_layers=config.rnn_num_layers,
                           bidirectional=config.rnn_bidirectional,
                           dropout=config.dropout_rate)

        self.fc = nn.Linear(in_features=config.hidden_size * 2,
                            out_features=1)

    # x: shape(batch_size, essay_len)
    def forward(self, x):
        embed_x = self.embedding(x)  # batch_size * essay_len * embedding_size

        rnn_out = self.rnn(embed_x)     # batch_size * essay_len * hidden_size
        rnn_out = rnn_out[0]
        rnn_out = torch.sum(rnn_out, dim=1)   # batch_size * hidden_size
        rnn_out = torch.mul(rnn_out, 1 / self.config.max_essay_len)

        out = self.fc(rnn_out)   # batch_size * 1
        out = torch.squeeze(out)
        out = torch.sigmoid(out)

        return out


class AttentionNN(nn.Module):
    def __init__(self, config):
        super(AttentionNN, self).__init__()
        self.dropout_rate = config.dropout_rate
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_size)

        self.attention_Q = nn.Linear(in_features=config.embedding_size,
                                     out_features=config.d_model)

        self.attention_K = nn.Linear(in_features=config.embedding_size,
                                     out_features=config.d_model)

        self.attention_V = nn.Linear(in_features=config.embedding_size,
                                     out_features=config.d_model)

        self.rnn = nn.LSTM(input_size=config.d_model,
                           hidden_size=config.hidden_size,
                           batch_first=True,
                           num_layers=3,
                           bidirectional=True)

        self.fc = nn.Linear(in_features=config.hidden_size * 2,
                            out_features=1)

    # x: shape(batch_size, essay_len)
    def forward(self, x):
        embed_x = self.embedding(x)  # batch_size * essay_len * embedding_size

        Q = self.attention_Q(embed_x)   # essay_len * d_model
        K = self.attention_K(embed_x)   # essay_len * d_model
        V = self.attention_V(embed_x)   # essay_len * d_model

        K = K.permute(0, 2, 1)
        QK = torch.bmm(Q, K)
        QK = torch.mul(QK, 1 / math.sqrt(self.config.d_model))  # batch_size * essay_len * essay_len

        soft_QK = torch.softmax(QK, dim=-1)
        attention = torch.bmm(soft_QK, V)   # batch_size * essay_len * d_model

        rnn_out = self.rnn(attention)  # batch_size * essay_len * hidden_size
        rnn_out = rnn_out[0]
        rnn_out = torch.sum(rnn_out, dim=1)  # batch_size * hidden_size
        rnn_out = torch.mul(rnn_out, 1 / self.config.max_essay_len)

        fc_out = self.fc(rnn_out)
        fc_out = torch.squeeze(fc_out)
        out = torch.sigmoid(fc_out)

        return out
