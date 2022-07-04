import csv
import torch
import pkuseg


# end of sentence
eos = ['。', '？', '！', '：', '，']


def analyse(filename):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        max_cnt = 0
        max_len = 0
        for row in reader:
            cnt = 0
            length = 0
            for char in row[2]:
                if char in eos:
                    cnt += 1
                    if length > max_len:
                        max_len = length
                        max_len_filename = row[1]
                    length = 0
                else:
                    length += 1
            if cnt > max_cnt:
                max_cnt = cnt
                max_cnt_filename = row[1]

        print('max_sentence_num: ', max_cnt, ' in ', max_cnt_filename, sep='')
        print('max_sentence_len:', max_len, ' in ', max_len_filename, sep='')

        return max_cnt, max_len


def build_vocab(filenames):
    vocab = dict()
    vocab_cnt = 0

    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                for word in eval(row[2]):
                    if word in eos:
                        continue
                    elif vocab.get(word, -1) == -1:
                        vocab_cnt += 1
                        vocab[word] = vocab_cnt

    return vocab, vocab_cnt


def divide_sentence(filename):
    vocab, vocab_size = build_vocab(filename)
    print(vocab_size)

    # max_sentence_num, max_sentence_len = analyse('clear.csv')
    max_sentence_num, max_sentence_len = 87, 200

    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)

        essays = []
        scores = []

        for row in reader:
            sentences = []
            sentence = []
            for word in eval(row[2]):
                if word in eos:
                    while len(sentence) < max_sentence_len:
                        sentence.append(0)
                    sentences.append(sentence)
                    sentence = []
                else:
                    sentence.append(vocab[word])
            if sentence:
                while len(sentence) < max_sentence_len:
                    sentence.append(0)
                sentences.append(sentence)
            while len(sentences) < max_sentence_num:
                sentences.append([0 for _ in range(max_sentence_len)])
            essays.append(sentences)
            scores.append(int(eval(row[4])) // 5)

        torch_essays = torch.tensor(essays)
        torch_scores = torch.tensor(scores)

        return torch_essays, torch_scores


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
                split_essay = seg.cut(essay)

                new_row = [split_essay, score]
                writer.writerow(new_row)
