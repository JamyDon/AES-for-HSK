import csv
import torch
import pkuseg
from config import config


eos = ['。', '？', '！', '：', '，']
data_path = '../data/'
raw_files = ['train.csv', 'value.csv', 'test.csv']
split_files = ['split_train.csv', 'split_value.csv', 'split_test.csv']


def analyse(filenames):
    max_len = 0

    for filename in filenames:
        with open(data_path + filename, encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                length = 0
                for word in eval(row[0]):
                    length += 1
                if length > max_len:
                    max_len = length

    print('max_essay_len:', max_len, sep='')

    return max_len


def vocab_process(min_retain_freq=-1):
    with open('../data/vocab.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        vocab = []

        if min_retain_freq == -1:
            for word in reader:
                vocab.append(word[0])

        else:
            for word in reader:
                if eval(word[1]) < min_retain_freq:
                    break
                else:
                    vocab.append(word[0])
            vocab.append('')

        print('vocab size: ', len(vocab))
        return vocab


def prepare_data(filename, essay_grade_num, use, min_retain_freq=-1):
    vocab = vocab_process(min_retain_freq)
    vocab_size = len(vocab)

    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)

        essays = []
        lengths = []
        scores = []

        for row in reader:
            essay = []
            length = 0
            for word in eval(row[0]):
                length += 1
                if word not in vocab:
                    essay.append(vocab_size + 1)
                else:
                    essay.append(vocab.index(word) + 1)

            lengths.append(length)

            while length < 554:
                length += 1
                essay.append(0)

            essays.append(essay)
            scores.append(int(eval(row[1]) // (100 // essay_grade_num)))

        torch_essays = torch.tensor(essays)
        torch_scores = torch.tensor(scores)

        assert use in [0, 1, 2]

        if use == 0:
            torch.save(torch_essays, './data/train_input.pt')
            torch.save(torch_scores, './data/train_label.pt')

        elif use == 1:
            torch.save(torch_essays, './data/value_input.pt')
            torch.save(torch_scores, './data/value_label.pt')

        else:
            torch.save(torch_essays, './data/test_input.pt')
            torch.save(torch_scores, './data/test_label.pt')


def prepare_all():
    prepare_data('../data/split_train.csv', config.essay_grade_num, 0, config.min_retain_freq)
    prepare_data('../data/split_value.csv', config.essay_grade_num, 1, config.min_retain_freq)
    prepare_data('../data/split_test.csv', config.essay_grade_num, 2, config.min_retain_freq)


def load_data():
    print('loading data...')
    train_input = torch.load('./data/train_input.pt')
    train_label = torch.load('./data/train_label.pt')
    value_input = torch.load('./data/value_input.pt')
    value_label = torch.load('./data/value_label.pt')
    test_input = torch.load('./data/test_input.pt')
    test_label = torch.load('./data/test_label.pt')

    return train_input, train_label, value_input, value_label, test_input, test_label
