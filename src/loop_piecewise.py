import random
import math
import csv
from unittest import skip
import numpy as np

def Piecewise(t, epsilon):
    C = (math.exp(epsilon/2)+1)/(math.exp(epsilon/2)-1)
    l = ((C+1)/2)*t - (C-1)/2
    r = l + C -1
    
    x = random.random()
    
    if x < math.exp(epsilon/2)/(math.exp(epsilon/2)+1):
        return random.uniform(l, r)
    else:
        rate = abs(l+C) /(abs(l+C)+abs(C-r))
        x = random.random()
        if x < rate:
            return random.uniform(-C, l)
        else:
            return random.uniform(r, C)

def normal(attrset,skipindex):
    attrParm = []
    n = len(attrset)
    #まずは変化パラメータを調べる
    for i in range(n):
        if i in skipindex:
            attrParm.append([0,0])
        else:
            mid = (max(attrset[i]) + min(attrset[i])) / 2
            wid = abs(max(attrset[i]) - min(attrset[i]))
            attrParm.append([mid,wid])
    #ここで標準化
    normalAttrset = []
    for i in range(n):
        attrrow = []
        if i in skipindex:
            normalAttrset.append(attrset[i])
            continue 
        for j in range (len(attrset[i])):
            if attrParm[i][1] == 0:
                attrrow.append(0)
            else:
                attrrow.append(((attrset[i][j]-attrParm[i][0])/attrParm[i][1])*2)
        normalAttrset.append(attrrow)
    return normalAttrset, attrParm    

def reverseNormal(normalAttrset, attrParm,skipindex):
    n = len(normalAttrset)
    originalAttrset = []
    for i in range(n): 
        attrrow = []
        if i in skipindex:
            originalAttrset.append(normalAttrset[i])
            continue
        for normalAttr in normalAttrset[i]:
            attrrow.append(normalAttr*attrParm[i][1]/2+attrParm[i][0])
        originalAttrset.append(attrrow)
    return originalAttrset

"""""""""""""""""""""    
Parameter Setting
"""""""""""""""""""""
# 生データのパス
openfile = r"C:\Users\yoshi\OneDrive\Documents\Lecture\Prosec\data\wdbc(k=5).csv"
# skipindex:説明変数以外のインデックス
skipindex = [0,1]
# "各属性に対する"εの値．全体でεm=10,今回は5属性なので1属性当たりε=10/5です
epsilon = 10/5
# ループ回数
loop = 1000
"""""""""""""""""""""    
Parameter Setting
"""""""""""""""""""""
with open(openfile) as file:
    reader = csv.reader(file)
    rowset = [row for row in reader]
n = len(rowset[0])
attrset = [[] for i in range(n)]
for row in rowset:
    for i in range(n):
        if i in skipindex:
            attrset[i].append(row[i])
        else:    
            attrset[i].append(float(row[i]))


for l in range(loop):
    createfile = r"../data_ionosphere/largeData/PM_breast_εm=2/ionosphere_PMfile" + str(l) + ".csv" 
    #標準化[a,b]->[-1,1]
    normalAttrset, attrParm = normal(attrset,skipindex)
    #PM
    PMattrset = []
    for i in range(n):
        if i in skipindex:
            PMattrset.append([s for s in normalAttrset[i]])
            continue
        PMattrset.append([Piecewise(s,epsilon) for s in normalAttrset[i]])
    #逆標準化[-1,1]->[a,b]
    original = reverseNormal(PMattrset,attrParm,skipindex)
    
    #配列を属性順からレコード順に転置
    original_T = np.array(original).T.tolist()
    #データ保存
    with open(createfile, 'w',newline="") as file:
            writer = csv.writer(file)
            writer.writerows(original_T)
    print(l)

