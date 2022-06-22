
#import Analyze
import csv
import Split
import numpy as np
import random
import math
import pandas as pd
import LDP
"""""""""""""""
Parameter Setting
"""""""""""""""

#相関係数を計算する
def cov(data):
    """
    df = pd.DataFrame(data)
    corr = df.corr()
    ind_lst = [0,0] + list(corr[1].to_numpy())[2:]
    for i in range(len(ind_lst)):
        if math.isnan(ind_lst[i]):
             ind_lst[i]= 0
    """
    l = len(data[0])
    n = len(data)
    data = np.array(data)
    ind_lst = [0 for i in range(l)]
    for i in range(3,l):
        x, y = [],[]
        for j in range(n):
            if data[j][i] ==None:
                continue
            x.append(data[j][1])
            y.append(data[j][i])
        ind_lst[i] = np.corrcoef(np.array(x),np.array(y))[0][1]        
    return(ind_lst)


def chose(lst,k):
    import heapq
    lst = [abs(x) for x in lst]
    bigk = heapq.nlargest(k,lst)
    num_lst = []
    for i in bigk:
        num_lst.append(lst.index(i))
    return([0,1,2] + sorted(num_lst)) 

#====================================================
#Step1: split all items in csv according to the range of each attribute.
#=======================================================================

#case1: split according to the same length of the entire range.
#splitset,midSet = Split.getSplitTupleSetForAttributes(openfile,skipindex,5,5)

#case2: split according to the same number of attributes.
"""
splitset, midSet = Split.getSplitTupleSetForAttributes2(openfile,skipindex,5)#タプルと中央値の作成
Split.makeLabeledFile(openfile,labeledfile,splitset,skipindex)#汎化ファイルの生成
"""

"""
#Step3: Analyze a csv made by LDP's output
#createfile: the location of an output of LDP
#analyzefile: the location of analysis output
Analyze.Analyze(createfile,analyzefile,analyzedglaph)

#Step4: Machine Learning without dimension reduction
#learning result by SVM 
result = Analyze.SVM(analyzefile,IDs)
print(result)
"""