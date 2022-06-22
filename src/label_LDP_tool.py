
#import Analyze
import csv
import Split
import numpy as np
import random
import math
import pandas as pd
import LDP_tool
import Math_tool
"""""""""""""""
Parameter Setting
"""""""""""""""
"属性の特性を定義"
#id
att_id = [0]
#ノイズ無し目的変数
att_obj = [1]
#ノイズ有り目的変数
att_noised_obj = [2]
#ノイズ有り説明変数
att_exp = list(range(3,33))

def label_LDP(labeledfile, labeledldpfile, epsilon_att, label_count):
    with open(labeledfile) as file:
        reader = csv.reader(file)
        rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    for i in range(len(rowset)):
        for j in range(len(rowset[0])):
            if j in att_exp:
                rowset[i][j] = LDP_tool.label(rowset[i][j], epsilon_att, label_count)#ひとつづつ考えていく
            if j in att_noised_obj:
                rowset[i][j] = LDP_tool.one_bit_LDP(rowset[i][j], epsilon_att)
    with open(labeledldpfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rowset)    
#print(midSet)
#=======================================================================
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