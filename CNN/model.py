# import torch
# from torch.autograd import Variable
# import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# from torchvision import datasets, transforms
# import os


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.is_training = True
        self.dropout_rate = config.dropout_rate
        self.use_element = config.use_element
        self.config = config

        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_size)

        self.word_conv = nn.Sequential(nn.Conv1d(in_channels=config.embedding_size,
                                                 out_channels=config.sentence_feature_size,
                                                 kernel_size=config.word_conv_size),
                                       nn.ReLU(),
                                       nn.AvgPool1d(kernel_size=config.max_sentence_len - config.word_conv_size + 1))

        self.sentence_conv = nn.Sequential(nn.Conv1d(in_channels=config.sentence_feature_size,
                                                     out_channels=config.essay_feature_size,
                                                     kernel_size=config.sentence_conv_size),
                                           nn.ReLU(),
                                           nn.AvgPool1d(
                                               kernel_size=config.max_sentence_num - config.sentence_conv_size + 1))

        self.fc = nn.Linear(in_features=config.sentence_feature_size,
                            out_features=1)

        # if os.path.exists(config.embedding_path) and config.is_training and config.is_pretrain:
        #     print("Loading pretrained embedding...")
        #     self.embedding.weight.data.copy_(torch.from_numpy(np.load(config.embedding_path)))

    # x: shape(batch_size, sentence_num, sentence_len)
    def forward(self, x):
        embed_x = self.embedding(x)  # batch_size * sentence_num * sentence_len * embedding_size
        embed_x = embed_x.permute(0, 1, 3, 2)  # batch_size * sentence_num * embedding_size * sentence_len

        sentence_out = self.word_conv(embed_x)  # batch_size * sentence_num * sentence_feature_size * 1
        sentence_out = sentence_out.view(-1, sentence_out.size(1))  # batch_size * sentence_num * sentence_feature_size
        sentence_out = sentence_out.permute(0, 2, 1)  # batch_size * sentence_feature_size * sentence_num

        essay_out = self.sentence_conv(sentence_out)  # batch_size * sentence_feature_size * 1
        essay_out = essay_out.view(-1, essay_out.size(1))  # batch_size * sentence_feature_size

        essay_out = F.dropout(input=essay_out, p=self.dropout_rate)
        essay_scores = self.fc(essay_out)  # batch_size

        return essay_scores
