import csv
import pkuseg


def read_raw(filename):
    with open(filename, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        seg = pkuseg.pkuseg()

        header = next(reader)

        index = []
        filename = []
        content = []
        title = []
        score = []

        for row in reader:
            index.append(row[0])
            filename.append(row[1])
            essay = row[2]
            content.append(seg.cut(essay))
            title.append(row[3])
            score.append(row[4])

        return content, score


def content2matrix(content):
    print(1)


read_raw("clear.csv")