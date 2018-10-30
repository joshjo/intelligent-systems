import csv


with open('data/train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    d = []
    for i, row in enumerate(csv_reader):
        x = []
        for j, col in enumerate(row):
            if i == 0 or j == 0:
                continue
            x.append(col if col.isdigit() else '0')
        d.append(' '.join(x))
    print('[', '; '.join(d), ']')
