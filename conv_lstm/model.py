import torch
import torch.nn as nn
import torch.nn.functional as F
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.lin = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, encoder_hidden_states, decoder_hidden_state):
        dh = self.lin(decoder_hidden_state).unsqueeze(-1)     # N * H * 1
        attn_scores = torch.bmm(encoder_hidden_states, dh)    # N * L * 1 注意力（相关性）分数
        weights = F.softmax(attn_scores, dim=1)               # 在L维度上归一化得到权重
        outputs = (weights * encoder_hidden_states).sum(1)    # N * H    在L维度上加权求和
        return outputs
class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.is_training = config.is_training
        self.dropout_rate = config.dropout_rate
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_size)
        self.CNN = nn.Conv1d(in_channels=config.embedding_size,out_channels=config.feature_size,kernel_size=3)
        self.RNN = nn.LSTM(input_size=config.feature_size,
                          hidden_size=config.hidden_size,
                          batch_first=True,
                          num_layers=2,
                          bidirectional=True,
                          dropout=config.dropout_rate)

        self.fc = nn.Linear(in_features=config.hidden_size*2,
                            out_features=config.essay_grade_num)
        self.sig=nn.Sigmoid()
    def forward(self, x):
        embed_x = self.embedding(x)
        embed_x = embed_x.permute(0,2,1)
        CNN_out = self.CNN(embed_x)
        CNN_out = CNN_out.permute(0,2,1)
        RNN_out = self.RNN(CNN_out)
        RNN_out = RNN_out[0]
        RNN_out = torch.sum(RNN_out, dim=1)
        RNN_out = torch.mul(RNN_out, 1 / self.config.max_essay_len)
        out = self.fc(RNN_out)
        return self.sig(out)
