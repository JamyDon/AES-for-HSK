import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.is_training = config.is_training
        self.dropout_rate = config.dropout_rate
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_size)

        self.rnn = nn.RNN(input_size=config.embedding_size,
                          hidden_size=config.hidden_size,
                          batch_first=True,
                          num_layers=2,
                          bidirectional=True,
                          dropout=config.dropout_rate)

        self.fc = nn.Linear(in_features=config.hidden_size * 2,
                            out_features=config.essay_grade_num)

    # x: shape(batch_size, essay_len)
    def forward(self, x):
        embed_x = self.embedding(x)  # batch_size * essay_len * embedding_size

        rnn_out = self.rnn(embed_x)     # batch_size * essay_len * hidden_size
        rnn_out = rnn_out[0]
        rnn_out = torch.sum(rnn_out, dim=1)   # batch_size * hidden_size
        rnn_out = torch.mul(rnn_out, 1 / self.config.max_essay_len)

        out = self.fc(rnn_out)   # batch_size * essay_grade_num

        return out
