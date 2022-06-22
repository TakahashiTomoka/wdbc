import csv
import math
import numpy as np

#maxnum = BFlength
#split according to the same length of the entire range.
def getSplitTupleSetForAttributes(filename, skipindex,minnum,maxnum):
    with open(filename) as file:
        reader = csv.reader(file)
        rowset = [row for row in reader]
    n = len(rowset[0])
    result = []
    minset = [float(100000) for i in range(n)]
    maxset = [float(0) for i in range(len(rowset[0]))]
    for row in rowset:
        for i in range(n):
            if i in skipindex:
                continue
            if float(row[i]) < minset[i]:
                minset[i] = float(row[i])
            if float(row[i]) > maxset[i]:
                maxset[i] = float(row[i])
    midResult = []
    for i in range(n):
        if i in skipindex:
            result.append([])
            continue
        #define the number of split
        length = math.floor(maxset[i]-minset[i])
        flength = maxset[i]-minset[i]
        splitTupleSet = []
        midSet = []
        if length <= minnum:
            m = minset[i]
            M = maxset[i]
            d = flength/minnum
            for j in range(minnum):
                if j!= minnum-1:
                    splitTupleSet.append((m,m+d))
                    midSet.append((m+(d/2)))
                    m += d
                else:
                    splitTupleSet.append((m,M))
                    midSet.append((m+M)/2)
        elif length > minnum:
            m = minset[i]
            M = maxset[i]
            num = min(length,maxnum)
            d = flength/float(num)
            for j in range(num):
                if j!= num-1:
                    splitTupleSet.append((m,m+d))
                    midSet.append((m+(d/2)))
                    m += d
                else:
                    splitTupleSet.append((m,M))
                    midSet.append((m+M)/2)
        result.append(splitTupleSet)
        midResult.append(midSet)
    return result, midResult

#split according to the same number of attributes.
def getSplitTupleSetForAttributes2(filename, skipindex, splitnum):
    with open(filename) as file:
        reader = csv.reader(file)
        rowset = [row for row in reader]
    n = len(rowset[0])
    l = len(rowset)
    attrset = [[] for i in range(n)]
    for row in rowset:
        for i in range(n):
            if row[i] != '':
                if i in skipindex:
                    attrset[i].append(row[i])
                else:
                    attrset[i].append(float(row[i]))
    #print(attrset)
    result = []
    midResult = []
    for i in range(n):
        rangeset = []
        # 説明変数以外
        if i in skipindex:
            result.append([])
            continue
        #候補数がラベル分割数未満の場合を考慮する場合，以下のコメントアウトを解除 
        """
        #候補数がラベル分割数未満(実質離散値)
        cand = list(set(attrset[i]))
        if len(cand)<splitnum:
            midSet = []
            for j in range(len(cand)):
                value = float(cand[j])
                d = 1/1000 # 入力値の差未満の微小値
                rangetuple = (value,value+d)
                midSet.append(value)
                rangeset.append(rangetuple)
            pad = 1024 #pad="dummy"でもいいけど解析側でエラー吐くから注意
            for j in range(splitnum-len(cand)):
                midSet.append(pad)
            result.append(rangeset)
            midResult.append(midSet)
            continue
        """
        sortedset = np.sort(attrset[i])
        midSet = []
        for j in range(splitnum):
            l = len(sortedset)
            a = float(sortedset[int(l*j/splitnum)])
            b = float(sortedset[min(int(l*(j+1)/splitnum),l-1)])
            rangetuple = (a,b)
            midSet.append((a+b)/2)
            rangeset.append(rangetuple)
        result.append(rangeset)
        midResult.append(midSet)
    #print(result, midResult)
    return result, midResult

#split according to the same number of attributes (continuous&discrete).
def getSplitTupleSetForAttributes_(filename, skipindex,continuousIndex,discreteIndex,splitnum):
    with open(filename) as file:
        reader = csv.reader(file)
        rowset = [row for row in reader]
    n = len(rowset[0])
    l = len(rowset)
    attrset = [[] for i in range(n)]
    for row in rowset:
        for i in range(n):
            if i in skipindex:
                attrset[i].append(row[i])
            elif i in continuousIndex:
                attrset[i].append(float(row[i]))
            elif i in discreteIndex:
                attrset[i].append(row[i])
    result = []
    midResult = []
    for i in range(n):
        rangeset = []
        if i in skipindex:
            result.append([])
            continue
        if i in continuousIndex:
            sortedset = np.sort(attrset[i])
            midSet = []
            for j in range(splitnum):
                a = float(sortedset[int(l*j/splitnum)])
                b = float(sortedset[min(int(l*(j+1)/splitnum),l-1)])
                rangetuple = (a,b)
                midSet.append((a+b)/2)
                rangeset.append(rangetuple)
            result.append(rangeset)
            midResult.append(midSet)
        elif i in discreteIndex:
            uniqueDiscreteSet = list(set(attrset[i]))
            result.append(uniqueDiscreteSet)
            midResult.append(uniqueDiscreteSet)
            
            
            
    return result, midResult

def isfloat(s):  # 浮動小数点数値を表しているかどうかを判定
    try:
        float(s)  # 文字列を実際にfloat関数で変換してみる
    except ValueError:
        return False
    else:
        return True

def Label(item,tupleSet):
    for tuple in tupleSet:
        if tuple[0] <= float(item) and float(item) <=tuple[1]:
            return str(tuple[0]) + str(":") + str(tuple[1])
    return "error"

def Label2(item,tupleSet):
    cnt = 0
    #連続値の場合
    if isfloat(item):
        for tuple in tupleSet:
            if tuple[0] <= float(item) and float(item) <tuple[1]:
                return cnt
            cnt += 1
        return cnt-1
    #離散値の場合
    else:
        for tuple in tupleSet:
            if tuple == item:
                return cnt
            cnt += 1
        
    return "error"

#splitTupleSetForAttributes: 連続値の場合は区間，離散値の場合はその値
def makeLabeledFile(filename, createfilename,splitTupleSetForAttributes,skipindex):
    with open(filename) as file:
        reader = csv.reader(file)
        rowset = [row for row in reader]
    result = []
    for row in rowset:
        labelset = []
        for i in range(len(row)):
            if i in skipindex:
                labelset.append(row[i])
            else:
                labelset.append(Label2(row[i],splitTupleSetForAttributes[i]))
        result.append(labelset)
    with open(createfilename, 'w',newline="") as cfile:
        writer = csv.writer(cfile)
        writer.writerows(result)
    
    
