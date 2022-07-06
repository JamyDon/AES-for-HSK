class config:
    epoch_num = 10
    batch_size = 100
    lr = 0.001
    dropout_rate = 0.5

    vocab_size = 41000
    min_retain_freq = -1
    max_essay_len = 554

    embedding_size = 128
    d_model = 128
    rnn_num_layers = 1
    rnn_bidirectional = True
    hidden_size = 128
    essay_grade_num = 20