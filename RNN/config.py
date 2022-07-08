class config:
    loss_first = True
    epoch_num = 20
    batch_size = 100
    lr = 0.001
    dropout_rate = 0.5

    vocab_size = 41000
    min_retain_freq = -1
    max_essay_len = 554

    embedding_size = 64
    d_model = 64
    rnn_num_layers = 1
    rnn_bidirectional = True
    hidden_size = 64
    essay_grade_num = 20
