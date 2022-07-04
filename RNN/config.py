class config:
    is_training = True
    epoch_num = 10
    batch_size = 100
    lr = 0.001
    dropout_rate = 0.5

    vocab_size = 41000
    min_retain_freq = -1
    max_essay_len = 554

    embedding_size = 64
    hidden_size = 64
    essay_grade_num = 20

    word_conv_size = 4
    sentence_conv_size = 2
