import csv

def reg(rowset):
    for i in range(len(rowset[0])):
        max_num = -100000
        min_num = 100000
        for j in range(len(rowset)):
            if rowset[j][i] > max_num:
                max_num = rowset[j][i]
            if rowset[j][i] < min_num:
                min_num = rowset[j][i]
        
        mid = (max_num + min_num)/2
        wide = (max_num - min_num)
        
        for j in range(len(rowset)):
            rowset[j][i] = ((rowset[j][i] - mid) / wide) * 2
    return rowset



openfile = r"C:\Users\u032721b\Documents\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\wdbc(num).csv"
regfile = r"C:\Users\u032721b\Documents\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\wdbc(num)_reg.csv"


with open(openfile) as file:
            reader = csv.reader(file)
            rowset = [row for row in reader]
rowset = [[float(v) for v in row] for row in rowset]

with open(regfile, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(reg(rowset))