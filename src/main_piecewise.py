import random
import math
import csv

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

def normal(attrset):
    attrParm = []
    n = len(attrset)
    #まずは変化パラメータを調べる
    for i in range(n):
        mid = (max(attrset[i]) + min(attrset[i])) / 2
        wid = abs(max(attrset[i]) - min(attrset[i]))
        attrParm.append([mid,wid])
    #ここで標準化
    normalAttrset = []
    for i in range(n):
        attrrow = []
        for j in range (len(attrset[i])):
            attrrow.append(((attrset[i][j]-attrParm[i][0])/attrParm[i][1])*2)
        normalAttrset.append(attrrow)
    return normalAttrset, attrParm    

def reverseNormal(normalAttrset, attrParm):
    n = len(normalAttrset)
    originalAttrset = []
    for i in range(n): 
        attrrow = []
        for normalAttr in normalAttrset[i]:
            attrrow.append(normalAttr*attrParm[i][1]/2+attrParm[i][0])
        originalAttrset.append(attrrow)
    return originalAttrset
            

openfile = r"../data_ionosphere/ionosphere(continuous)_5.csv" #生データ
with open(openfile) as file:
    reader = csv.reader(file)
    rowset = [row for row in reader]
n = len(rowset[0])
attrset = [[] for i in range(n)]
for row in rowset:
    for i in range(n):
        attrset[i].append(float(row[i]))

epsilon = 10
#標準化[a,b]->[-1,1]
normalAttrset, attrParm = normal(attrset)
#PM
PMattrset = []
for i in range(n):
    PMattrset.append([Piecewise(s,epsilon) for s in normalAttrset[i]])
#逆標準化[-1,1]->[a,b]
original = reverseNormal(PMattrset,attrParm)
print(original)
#print(normalAttrset)


"""
attrParm = []
for i in [0]:
    print(str(i)+"属性")
    print(max(attrset[i]))
    print(min(attrset[i]))
    mid = (float(max(attrset[i])) + float(min(attrset[i]))) / 2
    wid = abs(float(max(attrset[i])) - float(min(attrset[i])))
    attrParm.append([mid,wid])
sample = []
for attr in attrset[0]:
    sample.append(((attr-attrParm[0][0])/attrParm[0][1])*2)
print(max(sample))
print(min(sample))
#print(sample)
for s in sample:
    print(s*attrParm[0][1]/2+attrParm[0][0])
"""