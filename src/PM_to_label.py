# this is the program to analyze テスト=LDP　評価＝original or label
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

is_original = True

#files = glob.glob(r"C:\Users\takahasi tomoka\Prosec\largeData\(f=0.1,q=0.9,p=0.1)_要素数分割\*")
files = glob.glob(r"C:\Users\yoshi\OneDrive\Documents\Lecture\Prosec\data_Ionosphere\largeData\PM_breast_εm=2\*")
#files = glob.glob(r"C:\Users\takahasi tomoka\Desktop\新しいフォルダー\*")
#files = glob.glob(r"C:\Users\takahasi tomoka\Desktop\新しいフォルダー (2)\*")

if is_original:
    testdata = r"C:\Users\yoshi\OneDrive\Documents\Lecture\Prosec\data\wdbc(k=5).csv"
else:
    testdata =  r"C:\Users\yoshi\OneDrive\Documents\Lecture\Prosec\data_Ionosphere\ionosphere(continuous)_Labeledfile(coef).csv"

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



#main program
test_df = pd.read_csv(testdata,names = label)
"""
with open(to_numfile)as f:
    reader = csv.reader(f)
    To_num = [[float(v) for v in row] for row in reader]
To_num = np.array(To_num).T.tolist()
"""
with open(groupfile)as f:
    reader = csv.reader(f)
    crossed_ID = [[int(v) for v in row] for row in reader]
"""
if not is_original:
    #生データ空間へ
    for i in range(sample_num):
        test_df.iloc[i,2:] = get_label_to_num(test_df.iloc[i,2:],To_num,framesize)
"""
        
X_tests, y_tests = [],[]
for IDs in crossed_ID:
    X_test = []
    y_test = []
    for i in range(sample_num):
        if test_df.at[i,'ID'] not in IDs:
            X_test.append(test_df.iloc[i,2:])
            y_test.append(test_df.iloc[i,1])
    X_tests.append(X_test)
    y_tests.append(y_test)
    
start = time.time()
    
Scores = []
model = SVC(kernel='linear', random_state=None)

for traindata in files:
    train_df = pd.read_csv(traindata, names = label)
    
    #PMには不要
    """
    #生データ空間へ
    for i in range(sample_num):
        train_df.iloc[i,2:] = get_ldp_to_num(train_df.iloc[i,2:],To_num,framesize)
    """
    
    scores = []#1回分の結果
    for IDs, X_test, y_test in zip(crossed_ID,X_tests, y_tests):
        X_train = []
        y_train = []
        for i in range(sample_num):
            if train_df.at[i,'ID'] not in IDs:
                X_train.append(train_df.iloc[i,2:])
                y_train.append(train_df.iloc[i,1])
        #標準化　準備
        AVE, STD = [],[]            
        Ar_X_train = np.array(X_train)
        for i in range(feat_num):
            feat = Ar_X_train[:,i].tolist()
            ave, std = mean_std(feat)
            AVE.append(ave)
            STD.append(std)
            
        #標準化
        ST_X_train,ST_X_test = [],[]
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
        
        model.fit(ST_X_train,y_train)
        pred_test = model.predict(ST_X_test)
        score = accuracy_score(y_test, pred_test)
        scores.append(score)
    Scores.append(np.mean(scores))
    
    
#解析結果
print("実行時間：{}".format(time.time()-start))
print("平均：{}".format(np.mean(Scores)))
print("最高値：{}".format(max(Scores)))
print("最低値：{}".format(min(Scores)))
           
                     
    

    
    
