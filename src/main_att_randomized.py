from hashlib import new
import LDP
#import Analyze
import csv
import Split
import numpy as np
import random
import math
import pandas as pd
"""""""""""""""
Parameter Setting
"""""""""""""""


# f: a probability for randamization (keep almost the original BF)
# q: randomization of 1s in BF (q=1 -> keep 1, q=0 -> reversed to 0)
# p: randomization of 1s in BF (p=1 -> reversed to 1, p=0 -> keep 0)


f = 0.1
# 基本的にはq=1,p=0で固定．fはCalcEp.pyでεから逆算して求めると良いです．
q = 1
p = 0
# 説明変数以外のインデックス
skipindex = [0,1]
# この値は固定でOK
splitform = "要素数分割"
#属性数(30, or 5)
attrnum = 5


# 生データのパス（all_data） 

# 属性選択を行うためのノイズ付与データのパス
arrfile = r"C:\Users\u032721b\Documents\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\wdbc_num_noise1.csv"
# 生成した汎化データを保存するパス
labeledfile = r"C:\Users\u032721b\Documents\LDP\src\src\data\labelwdbc_5(1).csv"
# 生成した中央値データを保存するパス 
midfile = r"C:\Users\u032721b\Documents\LDP\src\src\data\Midfile_5(1).csv"
# 生成したLDPデータを保存するパス 
createfile = r"C:\Users\u032721b\Documents\LDP\src\src\data\test.csv"

#相関係数を計算する
def cov(data):
    df = pd.DataFrame(data)
    corr = df.corr()
    #print(corr)
    ind_lst = [0,0] + list(corr[1].to_numpy())[2:]
    for i in range(len(ind_lst)):
        if math.isnan(ind_lst[i]):
             ind_lst[i]= 0
    return(ind_lst)

def chose(lst,k):
    import heapq
    lst = [abs(x) for x in lst]
    bigk = heapq.nlargest(k,lst)
    num_lst = []
    for i in bigk:
        num_lst.append(lst.index(i))
    return([0,1] + sorted(num_lst)) 

#PWメカニズム
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

#一様ランダム
def random_uniform(t, epsilon):
    C = (math.exp(epsilon/2)+1)/(math.exp(epsilon/2)-1)
    return random.uniform(-C, C)

#データを[-1,1]になおすための処理
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

#第一段階で残す属性のランダム決定
def decide_att(sample_count, sample_len):
    l = list(range(2,sample_len))
    return random.sample(l, sample_count)

#第一段階(sample_countが残す属性数)
def random_att_del(rowset, sample_count, epsilon):
    sample_len = len(rowset[0])#要素数
    for i in range(len(rowset)):
        att = decide_att(sample_count, sample_len)#残す要素の決定
        rowset[i][0]=Piecewise(rowset[i][0], epsilon)#目的変数にPW
        for j in range(2, sample_len):
            
            if j not in att:
                rowset[i][j]=random_uniform(rowset[i][j], epsilon)#残さないものは一様ランダム(ここをNullに変更すれば残さないものはNullになる)
            else:
                rowset[i][j]=Piecewise(rowset[i][j], epsilon)#残すものにPW
    return rowset

#split according to the same number of attributes with random sampling.

#ノイズありでの属性削減（戻り値は選択された属性）
def noised_att_decide(rowset):

    
    sample_count = 30#残す属性数
    epsilon_all = 5 #全体でのイプシロン
    epsilon = epsilon_all/(sample_count + 1)#各属性のイプシロン
    sampleset = random_att_del(rowset, sample_count, epsilon)#
    #print(sampleset)
    #属性を決定
    c_lst = cov(sampleset)
    #print(c_lst)
    ind = chose(c_lst,attrnum)#IDのindex[0] + 目的変数のindex[1] +上位5属性のindex[, ,]
    #print(ind)
    return ind
    
"""""""""""""""
Parameter Setting
"""""""""""""""

if __name__ == "__main__":
    ind_len = []
    openfile = r"C:\Users\u032721b\Documents\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\wdbc(num)_reg.csv"#入力データ
    lenfile = r"C:\Users\u032721b\Documents\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\ind_len.csv"#決定された属性の保存先
    with open(openfile) as file:
            reader = csv.reader(file)
            rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    rowset = reg(rowset)#データの範囲を-1,1に
    c_lst = cov(rowset)
    #print(c_lst)
    print(chose(c_lst,attrnum))#ノイズなしの場合選択される属性を表示
    loop = 1000#試行回数
    for i in range(loop):#ノイズがある場合選択される属性を決定
        ind_len.append(noised_att_decide(rowset))
    with open(lenfile, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(ind_len)
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