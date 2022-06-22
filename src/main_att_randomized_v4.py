
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
labeledfile = r"C:\Users\u032721b\Documents\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\labelwdbc_5(1).csv"
# 生成した中央値データを保存するパス 
midfile = r"C:\Users\u032721b\Documents\LDP\src\src\data\Midfile_5(1).csv"
# 生成したLDPデータを保存するパス 

sample_count = 5#残す属性数
epsilon_all = 5 #全体でのイプシロン
epsilon = epsilon_all/(sample_count + 1)#各属性のイプシロン

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
    for i in range(2,l):
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
def random_uniform(t, epsilon0):
    C = (math.exp(epsilon0/2)+1)/(math.exp(epsilon0/2)-1)
    return random.uniform(-C, C)

#{-1,1}へのノイズ付与
def one_bit_LDP(t, epsilon0):
    C = (math.exp(epsilon0))/(1+math.exp(epsilon0))
    x = random.random()
    
    if x <= C:
        return t
    else:
        return -t

#データを[-1,1]になおすための処理
def reg(rowset):
    for i in range(1,len(rowset[0])):
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
def random_att_del(openfile, sample_count, epsilon0 = epsilon):
    with open(openfile) as file:
            reader = csv.reader(file)
            rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    rowset = reg(rowset)
    sample_len = len(rowset[0])#要素数
    for i in range(len(rowset)):
        att = decide_att(sample_count, sample_len)#残す要素の決定
        rowset[i][0]=one_bit_LDP(rowset[i][0], epsilon0)#目的変数にPW
        for j in range(2, sample_len):
            
            if j not in att:
                rowset[i][j] = None
                #rowset[i][j]=random_uniform(rowset[i][j], epsilon)#残さないものは一様ランダム(ここをNullに変更すれば残さないものはNullになる)
            else:
                rowset[i][j]=Piecewise(rowset[i][j], epsilon)#残すものにPW
    return rowset

#第一段階(ノイズ無し)
def random_att_del_no_noise(openfile, sample_count, epsilon0 = epsilon):
    with open(openfile) as file:
            reader = csv.reader(file)
            rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    rowset = reg(rowset)
    sample_len = len(rowset[0])#要素数
    for i in range(len(rowset)):
        att = decide_att(sample_count, sample_len)#残す要素の決定
        rowset[i][1]=one_bit_LDP(rowset[i][1], epsilon0)#目的変数にPW
        for j in range(2, sample_len):
            
            if j not in att:
                rowset[i][j] = None
                #rowset[i][j]=random_uniform(rowset[i][j], epsilon0)#残さないものは一様ランダム(ここをNullに変更すれば残さないものはNullになる)
    return rowset
#split according to the same number of attributes with random sampling.

def no_del_noise(openfile, fullnoisefile, epsilon0 = epsilon):
    with open(openfile) as file:
        reader = csv.reader(file)
        rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    rowset = reg(rowset)
    sample_len = len(rowset[0])#要素数
    for i in range(len(rowset)):
        rowset[i][1]=one_bit_LDP(rowset[i][1], epsilon0)
        for j in range(2, sample_len):
            rowset[i][j]=Piecewise(rowset[i][j], epsilon0)
    with open(fullnoisefile, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(rowset)
    
#ノイズなしでの属性削減（戻り値は選択された属性）
def unnoised_att_decide(openfile):
    sampleset = random_att_del_no_noise(openfile, sample_count, epsilon)#
    #print(sampleset)
    #属性を決定
    c_lst = cov(sampleset)
    #print(c_lst)
    ind = chose(c_lst,attrnum)#IDのindex[0] + 目的変数のindex[1] +上位5属性のindex[, ,]
    #print(ind)
    return ind, sampleset        
#ノイズありでの属性削減（戻り値は選択された属性）
def noised_att_decide(openfile):
    sampleset = random_att_del(openfile, sample_count, epsilon)#
    #print(sampleset)
    #属性を決定
    c_lst = cov(sampleset)
    #print(c_lst)
    ind = chose(c_lst,attrnum)#IDのindex[0] + 目的変数のindex[1] +上位5属性のindex[, ,]
    #print(ind)
    return ind, sampleset
    
def Split_test(openfile, deffile, createfile):
    splitset, midSet = Split.getSplitTupleSetForAttributes2(openfile,skipindex,5)
    #print(splitset)
    print(midSet)
    Split.makeLabeledFile(deffile,labeledfile,splitset,skipindex)
    splitnum = [len(splitTuple) for splitTuple in splitset]
    with open(labeledfile) as file:
            reader = csv.reader(file)
            rowset = [row for row in reader]
    LDPset = []
    for row in rowset:
        newrow = []
        for i in range(len(row)):
            if i in skipindex:
                newrow.append(row[i])
                continue    
            """
            #case1: construction BF
            ItemSet = LDP.BloomFilter(k,m,salts)
            ItemSet.setBF(str(row[i])) 
            """
                
            #case2: construction ItemBox
            ItemSet = LDP.ItemBox(splitnum[i])
            if not ((row[i] is None) or (row[i] == '')):
                ItemSet.setItem(int(row[i])) 
                
                
            S = LDP.LDP("ItemBox",ItemSet,f,q,p)
            #convert a LDP output (array) to one string
            newitem = "".join(list(map(str,S)))
            newrow.append(float(newitem))
        LDPset.append(newrow)
        
    with open(createfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(LDPset)

#ノイズありでの試行
def noise_loop():
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
        ind_len.append(noised_att_decide(openfile))
    with open(lenfile, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(ind_len)
        
def noise_split():
    
    
    openfile = r"C:\Users\u032721b\Documents\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\wdbc(num).csv"#入力データ
   
    createfile = r"C:\Users\u032721b\Documents\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\split_noise.csv"
    noisefile = r"C:\Users\u032721b\Documents\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\noise_len.csv"
    fullnoisefile =  r"C:\Users\u032721b\Documents\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\full_noise_len.csv"
    with open(openfile) as file:
            reader = csv.reader(file)
            rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    rowset = reg(rowset)#データの範囲を-1,1に
    c_lst = cov(rowset)
    #print(c_lst)
    print(chose(c_lst,attrnum))#ノイズなしの場合選択される属性を表示
    #ノイズがある場合選択される属性を決定
    ind, sampleset = noised_att_decide(openfile)
    print(ind)
    with open(noisefile, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(sampleset)
    no_del_noise(openfile, fullnoisefile)
    Split_test(noisefile, fullnoisefile, createfile)
    
def PW_loop(data):
    epsilon0 = epsilon
    #print(data)
    data[1] = one_bit_LDP(data[1], epsilon0)
    for i in range(2,len(data)):
        data[i] = Piecewise(data[i], epsilon0)
    return data
    
def no_noise_split():
    
    
    openfile = r"F:\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\wdbc(num)_reg.csv"#入力データ
   
    createfile = r"F:\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\split_no_noise.csv"
    noisefile = r"F:\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\no_noise_len.csv"
    fullnoisefile =  r"F:\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\full_noise_len.csv"
    with open(openfile) as file:
            reader = csv.reader(file)
            rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    rowset = reg(rowset)#データの範囲を-1,1に
    c_lst = cov(rowset)
    #print(c_lst)
    print(chose(c_lst,attrnum))#ノイズなしの場合選択される属性を表示
    #ノイズがある場合選択される属性を決定
    ind, sampleset = unnoised_att_decide(openfile)
    print(ind)
    with open(noisefile, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(sampleset)



def select_ldp():
    openfile = r"F:\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\wdbc(num)_reg.csv"#入力データ
   
    createfile = r"F:\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\split_no_noise.csv"
    noisefile = r"F:\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\noise_len.csv"
    fullnoisefile =  r"F:\LDP\LDP引継ぎ資料\LDP引継ぎ資料\data\breast\full_noise_len.csv"
    with open(openfile) as file:
            reader = csv.reader(file)
            rowset = [row for row in reader]
    rowset = [[float(v) for v in row] for row in rowset]
    rowset = reg(rowset)#データの範囲を-1,1に
    #print(rowset)
    c_lst = cov(rowset)
    #print(c_lst)
    print(chose(c_lst,attrnum))#ノイズなしの場合選択される属性を表示
    #ノイズがある場合選択される属性を決定
    ind, sampleset = noised_att_decide(openfile)
    print(ind)
    
    #Split_test(noisefile, openfile, createfile)
    rowset = [PW_loop(rowset[i]) for i in range(len(rowset))]

    with open(fullnoisefile, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerows(rowset)
    
    
    
if __name__ == "__main__":
    select_ldp()
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