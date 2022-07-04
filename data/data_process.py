import csv
import pkuseg


eos = ['。', '？', '！', '：', '，']
raw_files = ['train.csv', 'value.csv', 'test.csv']
split_files = ['split_train.csv', 'split_value.csv', 'split_test.csv']


def split(filename):
    seg = pkuseg.pkuseg()

    with open(filename, encoding='utf-8') as f:
        new_filename = 'split_' + filename
        with open(new_filename, mode='a', newline='', encoding='utf-8') as nf:
            writer = csv.writer(nf)
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                essay = row[1]
                score = row[3]

                if score == '':
                    continue

                split_essay = seg.cut(essay)

                new_row = [split_essay, score]
                writer.writerow(new_row)


def split_all():
    for file in raw_files:
        split(file)


def build_vocab(filenames):
    vocab = dict()
    vocab_cnt = 0

    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f)

            for row in reader:
                for word in eval(row[0]):
                    if word in eos:
                        continue
                    elif vocab.get(word, -1) == -1:
                        vocab_cnt += 1
                        vocab[word] = 1
                    else:
                        vocab[word] += 1

    sorted_vocab = sorted(vocab.items(),  key=lambda d: d[1], reverse=True)
    with open('vocab.csv', mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        for word in sorted_vocab:
            new_row = [word[0], word[1]]
            writer.writerow(new_row)
