class Config:
    is_training = True
    epoch_num = 10
    batch_size = 100
    lr = 0.001
    dropout_rate = 0.3

    vocab_size = 36000
    max_sentence_len = 200
    max_sentence_num = 87

    embedding_size = 32
    sentence_feature_size = 32
    essay_feature_size = 32
    essay_grade_num = 10

    word_conv_size = 5
    sentence_conv_size = 3
