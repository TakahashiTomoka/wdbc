# this is the program to analyze テスト=LDP　評価＝LDP
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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

framesize  =5
sample_num = 569
feat_num = 5 #30

#original data
#files = glob.glob(r"C:\Users\takahasi tomoka\Prosec\largeData\(f=0.1,q=0.9,p=0.1)_要素数分割\*")
files = glob.glob(r"C:\Users\yoshi\OneDrive\Documents\Lecture\Prosec\data_Ionosphere\largeData\PM_breast_εm=2\*")
#files = glob.glob(r"C:\Users\takahasi tomoka\Desktop\新しいフォルダー\*")
#files = glob.glob(r"C:\Users\takahasi tomoka\Desktop\新しいフォルダー (2)\*")



groupfile = r"C:\Users\yoshi\OneDrive\Documents\Lecture\Prosec\data\crossd_ID.csv"
#to_numfile = r"C:\Users\yoshi\OneDrive\Documents\Lecture\Prosec\data_Ionosphere\ionosphere(continuous)_Midfile(coef).csv"




label = ['ID',
         'revue',
         
         'mean radius',
         'mean texture', 
         'mean perimeter',
         'mean area',
         'mean smoothness',
         """
         'mean compactness',
         'mean concavity',
         'mean concave points',
         'mean symmetry',
         'mean fractal dimension',

         'radius SE',
         'texture SE', 
         'perimeter SE',
         'area SE',
         'smoothness SE',
         'compactness SE',
         'concavity SE',
         'concave points',
         'symmetry SE',
         'fractal dimension SE',

         'worst radius',
         'worst texture', 
         'worst perimeter',
         'worst area',
         'worst smoothness',
         'worst compactness',
         'worst concavity',
         'worst concave points',
         'worst symmetry',
         'worst fractal dimension'
         """
            ]




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

"""
#int -> label -> 数値 
def get_label_to_num(lst,Table,framesize):#lstの成分はc #tableはリスト値から整数値への変換
    LST = []
    for  num,table in zip(lst,Table):
        LST.append(table[int(num)])
    return(LST)



#int -> label -> 数値 
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
"""
  
#データセットの学習
def CSV(ldpdata, groups):
    # データのロード
    df = pd.read_csv(ldpdata,names = label)
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    """
    with open(to_num)as f:
        reader = csv.reader(f)
        To_num = [[float(v) for v in row] for row in reader]
    To_num = np.array(To_num).T.tolist()
    #生データ空間へ
    for i in range(sample_num):
        df.iloc[i,2:] = get_ldp_to_num(df.iloc[i,2:],To_num,framesize)
    """
   
    scores = []
    # 線形SVMのインスタンスを生成
    model = SVC(kernel='linear', random_state=None)
    for IDs in crossed_ID:
        # データの分割  
        X_train = []#data
        y_train = []#target
        X_test = []#data
        y_test = []#target
        for i in range(sample_num):
            if df.at[i,'ID'] in IDs:
                X_test.append(df.iloc[i,2:])
                y_test.append(df.iloc[i,1])
            else:
                X_train.append(df.iloc[i,2:])
                y_train.append(df.iloc[i,1])
        AVE = []
        STD = []
        Ar_X_train = np.array(X_train)
        for i in range(5):#30
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
        score = accuracy_score(y_test, pred_test)
        scores.append(score)    
    return(np.mean(scores))

start = time.time()
Scores = []
for file in files:
    Scores.append(CSV(file,groupfile))
    
    


    
    
#解析結果
print("実行時間：{}".format(time.time()-start))
print("平均：{}".format(np.mean(Scores)))
print("最高値：{}".format(max(Scores)))
print("最低値：{}".format(min(Scores)))
           
                
        
            


