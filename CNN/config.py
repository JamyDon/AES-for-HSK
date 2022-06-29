class Config:
    is_training = True
    epoch_num = 100
    batch_size = 10
    lr = 0.001
    dropout_rate = 0.3

    vocab_size = 32000
    max_sentence_len = 100
    max_sentence_num = 55

    embedding_size = 64
    sentence_feature_size = 64
    essay_feature_size = 64

    word_conv_size = 5
    sentence_conv_size = 3
