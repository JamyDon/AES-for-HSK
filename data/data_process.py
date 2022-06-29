import csv


# end of sentence
eos = ['。', '？', '！']


def max_sentence_num():
    with open('clear.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        max_cnt = 0
        for row in reader:
            cnt = 0
            for char in row[2]:
                if char in eos:
                    cnt += 1
            if cnt > max_cnt:
                max_cnt = cnt
                max_filename = row[1]

        print('max_sentence_num: ', max_cnt, " in ", max_filename, sep='')


def divide_sentence(filename):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        essays = []

        for row in reader:
            sentences = []
            sentence = []
            for word in eval(row[2]):
                if word in eos:
                    sentences.append(sentence)
                    sentence = []
                else:
                    sentence.append(word)
            if sentence:
                sentences.append(sentence)
            essays.append(sentences)

        print(essays[0])
        return essays
