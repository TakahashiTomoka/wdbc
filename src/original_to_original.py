# this is the program to analyze テスト=original　評価＝original

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score
import csv
import numpy as np
from statistics import mean, median,variance,stdev
import scipy.stats
import statistics
import pandas as pd
import array
import random
import os
import glob
import time

framesize  =5 #次元削減後の大きさ
sample_num = 569 #569,351
feat_num = 31 #5,30,34 #総次元数

#original_file = r"../data/new_wdbc.csv" #目的変数が{0 = M, 1 = B}
original_file =r"C:\Users\u032721b\Documents\LDP\myfolder\data\wdbc.csv" 
#original data
files = glob.glob(r"../data/original/*")
#Label
#files = glob.glob(r"../data/label/*")
#RAPPOR
#files = glob.glob(r"../data/RAPPOR/*")
#PM
#files = glob.glob(r"../data/PM/*")


#-------------
#e=5
#files = glob.glob(r"../data/e=5/files/*")
to_numfile = r"../data/e=5/Midfile_5(1).csv"


#e=50
#files = glob.glob(r"../data/e=50/files/*")
#to_numfile = r"../data/e=50/Midfile_50(1).csv"



#------------

#cross-validation
groupfile = r"../data/crossd_ID.csv"




# 属性数に合わせて調節(よくコンマ忘れ/コンマ過多でエラーになるので注意)
#インデックス番号で属性の判別を行う
label = [i for i in range(feat_num + 2)]
#index = [0, 1, 12, 18, 24, 25, 30] #Label
#index = [0, 1, 4, 6, 7, 10, 25] #RAPPOR
#index = [0, 1, 3, 22, 25, 28, 29] #PM
#index = [0, 1, 2, 5, 6, 9, 12]#e=5
#index = [0, 1, 2, 9, 24, 25, 29] #e=50

def isfloat(s):  # 浮動小数点数値を表しているかどうかを判定
    try:
        float(s)  # 文字列を実際にfloat関数で変換してみる
    except ValueError:
        return False
    else:
        return True

def mean_std(lst):
    ave = mean(lst)#平均
    std = np.std(lst)#標準偏差 standard devitation
    return(ave, std)


#int -> label -> 数値 
def get_label_to_num(lst,Table,framesize):#lstの成分はc #tableはリスト値から整数値への変換
    LST = []
    for  num,table in zip(lst,Table):
        LST.append(table[int(num)])
    return(LST)



#int -> label -> 数値 
# [0,0,1,1,1]みたいな感じになってるもの
def get_ldp_to_num(lst,Table,framesize):#lstの成分はc #tableはリスト値から整数値への変換
    LST = []
    for  num,table in zip(lst,Table):
        counter = 0
        candi = []
        while num >0:
            counter +=1
            if num%10 !=0:
                candi.append(framesize-counter)
            num = num//10
        if len(candi) == 0:
            candi.append(random.randint(0,framesize-1))
        labelnum = random.choice(candi)
        LST.append(table[labelnum])
    return(LST)

  
#データセットの学習
def CSV(ldpdata, to_num, groups, index_file):
    #print(label)
    all_df = pd.read_csv(ldpdata,header=None,names=label)
    #print(all_df)
    original_df = pd.read_csv(original_file,header=None,names=label)#add
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    with open(index_file) as file:
        reader = csv.reader(file)
        index = [e for e in reader]
    index = np.array(index).reshape(-1)
    index = [int(i) for i in index]
    # データのロード
    """
#中央値に戻さない場合、コメントアウトを削除
    with open(to_num)as f:
        reader = csv.reader(f)
        To_num = [[float(v) for v in row] for row in reader]
        
    
    
    #生データ空間へ
    for i in range(sample_num):
        O_df.iloc[i,2:] = get_label_to_num(O_df.iloc[i,2:],To_num,framesize)
    
   """
    all_df.iloc[:,0] = original_df.iloc[:,0]#IDを正しいものに変更   
    df = all_df.iloc[:,index]#必要な属性だけに削減
    print(df)
    scores = [] # accuracy, recall,precision
    model = SVC(kernel='linear', random_state=None)# 線形SVMのインスタンスを生成
    for IDs in crossed_ID:
        # データの分割  
        X_train = []#data
        y_train = []#target
        X_test = []#data
        y_test = []#target
        for i in range(sample_num):
            if df.iat[i,0] in IDs:
                X_test.append(df.iloc[i,2:])
                y_test.append(df.iloc[i,1])
            else:
                X_train.append(df.iloc[i,2:])
                y_train.append(original_df.iloc[i,1])#ノイズの乗っていない目的変数を獲得
        AVE = []
        STD = []
        Ar_X_train = np.array(X_train)
        for i in range(framesize):#30
            feat = Ar_X_train[:,i].tolist()
            ave, std = mean_std(feat)
            AVE.append(ave)
            STD.append(std)
        #標準化
        ST_X_test, ST_X_train = [],[]
        #学習データの標準化
        for data in X_train:
            std = []
            for d,a,s in zip(data, AVE,STD):
                if s==0:
                    std.append(d)
                else:
                    std.append((d-a)/s)
            ST_X_train.append(std)
        #評価データの標準化
        for data in X_test:
            std = []
            for d,a,s in zip(data, AVE,STD):
                if s==0:
                    std.append(d)
                else:
                    std.append((d-a)/s)
            ST_X_test.append(std)
        #モデルの学習
        model.fit(ST_X_train,y_train)                                                  
        # テストデータに対する精度
        pred_test = model.predict(ST_X_test)
        score =[accuracy_score(y_test, pred_test),recall_score(y_test, pred_test),precision_score(y_test, pred_test)]
        scores.append(score)    
    return(np.mean(scores,axis=0)) # accuracy,recall,precisionの平均

start = time.time()
Scores = []
for file in files:
    Scores.append(CSV(file,to_numfile,groupfile))
#    Scores.append(CSV(file,groupfile))
    
    


    
    
#解析結果
print("実行時間：{}".format(time.time()-start))
print("平均：{}".format(np.mean(Scores ,axis=0)))
#print("最高値：{}".format(Scores.max(axis=0)))
#print("最低値：{}".format(Scores.min(axis=0)))
           
                
        
            


