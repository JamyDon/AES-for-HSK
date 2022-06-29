class Config:
    is_training = True
    epoch_num = 100
    batch_size = 10
    lr = 0.001
    dropout_rate = 0.3

    vocab_size = 35185
    max_sentence_len = 200
    max_sentence_num = 87

    embedding_size = 8
    sentence_feature_size = 8
    essay_feature_size = 8

    word_conv_size = 5
    sentence_conv_size = 3
