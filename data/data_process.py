import csv


comma = ['。', '？', '！']


def max_sentence_num():
    with open('clear.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        max_cnt = 0
        for row in reader:
            cnt = 0
            for char in row[2]:
                if char in comma:
                    cnt += 1
            if cnt > max_cnt:
                max_cnt = cnt
                max_filename = row[1]

        print('max_sentence_num: ', max_cnt, " in ", max_filename, sep='')
