class Config:
    is_training = True
    epoch_num = 5
    batch_size = 100
    lr = 0.001
    dropout_rate = 0.5

    vocab_size = 36000
    max_sentence_len = 200
    max_sentence_num = 87

    embedding_size = 16
    sentence_feature_size = 16
    essay_feature_size = 16
    essay_grade_num = 20

    word_conv_size = 4
    sentence_conv_size = 2
