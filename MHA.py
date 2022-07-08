import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MHA(nn.Module):
    def __init__(self,config):
        super(MHA, self).__init__()
        self.config=config
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,embedding_dim=config.embedding_size)
        self.attention_q=nn.Linear(in_features=config.embedding_size,out_features=config.model_dim)
        self.attention_k=nn.Linear(in_features=config.embedding_size,out_features=config.model_dim)
        self.attention_v=nn.Linear(in_features=config.embedding_size,out_features=config.model_dim)
        self.dim_per_head = config.model_dim // config.num_heads
        self.num_heads = config.num_heads
        self.linear_k = nn.Linear(config.model_dim, self.dim_per_head * config.num_heads)
        self.linear_v = nn.Linear(config.model_dim, self.dim_per_head * config.num_heads)
        self.linear_q = nn.Linear(config.model_dim, self.dim_per_head * config.num_heads)
        self.linear = nn.Linear(config.model_dim, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.model_dim)
        self.RNN = nn.LSTM(input_size=config.model_dim,
                          hidden_size=config.hidden_size,
                          batch_first=True,
                          num_layers=2,
                          bidirectional=True,
                          dropout=config.dropout)
        self.fc = nn.Linear(in_features=config.hidden_size*2,
                            out_features=config.essay_grade_num)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=-1)
    def forward(self,x):
        embed_x = self.embedding(x)
        key = self.attention_k(embed_x)
        value = self.attention_v(embed_x)
        query = self.attention_q(embed_x)

        res = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)
        scale = 1 / math.sqrt(self.config.model_dim)
        attention = torch.bmm(query, key.transpose(1, 2))
        attention = attention * scale
        attention = self.soft(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, value)
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        output = self.linear(context)
        output = self.dropout(output)
        output = self.layer_norm(res + output)
        RNN_out = self.RNN(output)
        RNN_out = RNN_out[0]
        RNN_out = torch.sum(RNN_out, dim=1)
        RNN_out = torch.mul(RNN_out, 1 / self.config.max_essay_len)
        out = self.fc(RNN_out)
        return self.sig(out)