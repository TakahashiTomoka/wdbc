
#import Analyze
import csv
import Split
import numpy as np
import random
import math
import pandas as pd
import LDP
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

#データを[-1,1]になおすための処理
def reg(startfile, regfile, att_exp):
    with open(startfile) as file:
        reader = csv.reader(file)
        rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    for i in att_exp:
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
    
    with open(regfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rowset)

#第一段階で残す属性のランダム決定
def random_att(sample_count, att_exp):
    return random.sample(att_exp, sample_count)

#属性のランダム削除(元ファイル,保存ファイル,説明変数,残す属性数)
def random_att_del(openfile, sampledfile, att_exp, sample_count):
    with open(openfile) as file:
        reader = csv.reader(file)
        rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    
    #属性削減
    for i in range(len(rowset)):
        att = random_att(sample_count, att_exp)#残す要素の決定
        not_del_att = att_id + att_obj + att_noised_obj + att
        for j in range(len(rowset[0])):
            if j not in not_del_att:
                rowset[i][j] = None
                
    with open(sampledfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rowset)
    
def att_decide(sampledfile, attfile, sample_count):
    with open(sampledfile) as file:
        reader = csv.reader(file)
        rowset = [row for row in reader]
    rowset = [[float(v) if v != '' else None for v in row] for row in rowset]   

    c_lst = Math_tool.cov(rowset)
    att = Math_tool.chose(c_lst,sample_count)
    with open(attfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerow(att)    
        
def labeling(regfile, sampledfile, labeledfile, midfile, label_count):
    skipindex = att_id+att_obj+att_noised_obj
    splitset, midSet = Split.getSplitTupleSetForAttributes2(sampledfile, skipindex, label_count)
    with open(midfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(midSet)
    Split.makeLabeledFile(regfile,labeledfile,splitset,skipindex)
    #print(midSet)
    
#区間で等分する関数。    
def simplabel(num, label_count):
    for i in range(label_count):
        if num <= (2/label_count)*(i+1) -1:
            return i

    return label_count-1    


def simplabel_del(openfile, sampledfile, midfile, att_exp, sample_count, label_count):
    with open(openfile) as file:
        reader = csv.reader(file)
        rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    
    #属性削減
    for i in range(len(rowset)):
        att = random_att(sample_count, att_exp)#残す要素の決定
        not_del_att = att_id + att_obj + att_noised_obj + att
        for j in range(len(rowset[0])):
            if j not in not_del_att:
                rowset[i][j] = None
            elif j in att_exp:
                rowset[i][j] = simplabel(rowset[i][j], label_count)
    make_midfile(midfile, label_count)            
    with open(sampledfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rowset)
        
def simplabel_all(openfile, sampledfile, midfile, att_exp, label_count):
    with open(openfile) as file:
        reader = csv.reader(file)
        rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    for i in range(len(rowset)):
        for j in range(len(rowset[0])):
            if j in att_exp:
                rowset[i][j] = simplabel(rowset[i][j], label_count)
    make_midfile(midfile, label_count)        
    with open(sampledfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rowset)
        
def make_midfile(midfile, label_count):
    before = -1
    mid_len = []
    for i in range(label_count):
        #print(before,(2/label_count)*(i+1)-1)
        mid_len.append(((2/label_count)*(i+1)-1 + before)/2)
        before = (2/label_count)*(i+1)-1
    mid_len = [mid_len for i in range(30)]#説明変数分、同じもので作成
    with open(midfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(mid_len)
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