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


def reshape(labeledfile, attfile):
    #属性読み込み
    with open(attfile) as file:
        reader = csv.reader(file)
        att = [row for row in reader]
    att = [int(v) for v in att[0]]    
    #選択属性のみのファイルに書き直し
    with open(labeledfile) as file:
        reader = csv.reader(file)
        rowset = [[row[i] for i in att] for row in reader]
    with open(labeledfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rowset)
        
def reshape_datasets(folder,attfile):
    for file in folder:
        reshape(file,attfile)
        
#-------------------


#中間値データのreshape        
def reshape_midfile(midfile, attfile):
    #属性読み込み
    with open(attfile) as file:
        reader = csv.reader(file)
        att = [row for row in reader]
    att = [int(v)-3 for v in att[0]]    
    #選択属性のみのファイルに書き直し
    with open(midfile) as file:
        reader = csv.reader(file)
        rowset = [[v for v in row] for row in reader]
    midset = []
    for i in att[3:]:
        midset.append(rowset[i])
    with open(midfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(midset)


# 属性数に合わせて調節(よくコンマ忘れ/コンマ過多でエラーになるので注意)
#インデックス番号で属性の判別を行う
label = [i for i in range(framesize + 3)]


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



#---------------------

#学習データ・評価データ：生データ
def SVM(original_data, groups):
    df = pd.read_csv(original_data,header=None,names=label)
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]

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
                X_test.append(df.iloc[i,3:])
                y_test.append(df.iloc[i,2])#ノイズ付き目的変数
            else:
                X_train.append(df.iloc[i,3:])
                y_train.append(df.iloc[i,1])#ノイズなし目的変数
        AVE = []
        STD = []
        Ar_X_train = np.array(X_train)
        for i in range(framesize):#5
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


#-----------------
#WA
#-------------
#学習データ・評価データ：WA or WADP (WA or WADPファイル、中央値データ、交差検定用データ)
def WA_SVM(WA, to_num, groups):
    df = pd.read_csv(WA,header=None,names=label)
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    #Representative 
    with open(to_num)as f:
        reader = csv.reader(f)
        To_num = [[float(v) for v in row] for row in reader]
    #生データ空間へ
    for i in range(sample_num):
        df.iloc[i,3:] = get_label_to_num(df.iloc[i,3:],To_num,framesize)
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
                X_test.append(df.iloc[i,3:])
                y_test.append(df.iloc[i,2])#ノイズ付き目的変数
            else:
                X_train.append(df.iloc[i,3:])
                y_train.append(df.iloc[i,1])#ノイズなし目的変数
        AVE = []
        STD = []
        Ar_X_train = np.array(X_train)
        for i in range(framesize):#5
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

def WA_PW_SVM(WA,PW, to_num, groups):
    train_df = pd.read_csv(WA,header=None,names=label)
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    #Representative 
    with open(to_num)as f:
        reader = csv.reader(f)
        To_num = [[float(v) for v in row] for row in reader]
    #生データ空間へ
    for i in range(sample_num):
        train_df.iloc[i,3:] = get_label_to_num(train_df.iloc[i,3:],To_num,framesize)
    
    models, AVEs,STDs = [],[],[]
    for IDs in crossed_ID:
        # データの分割  
        X_train = []#data
        y_train = []#target
        for i in range(sample_num):
            if train_df.iat[i,0] not in IDs:    
                X_train.append(train_df.iloc[i,3:])
                y_train.append(train_df.iloc[i,1])#ノイズなし目的変数
        AVE = []
        STD = []
        Ar_X_train = np.array(X_train)
        for i in range(framesize):#5
            feat = Ar_X_train[:,i].tolist()
            ave, std = mean_std(feat)
            AVE.append(ave)
            STD.append(std)
        AVEs.append(AVE)
        STDs.append(STD)
        #標準化
        ST_X_train = []
        #学習データの標準化
        for data in X_train:
            std = []
            for d,a,s in zip(data, AVE,STD):
                if s==0:
                    std.append(d)
                else:
                    std.append((d-a)/s)
            ST_X_train.append(std)
        #モデルの学習
        model = SVC(kernel='linear', random_state=None)
        model.fit(ST_X_train,y_train)
        models.append(model)                                                  
    Scores = [] # accuracy, recall,precision
    for pw in PW:
        test_df = pd.read_csv(pw,header = None,names= label)
        scores = []
        for IDs,model,AVE,STD in zip(crossed_ID,models,AVEs,STDs):
            X_test = []
            y_test = []
            for i in range(sample_num):
                if test_df.iat[i,0] in IDs:
                    X_test.append(test_df.iloc[i,3:])
                    y_test.append(test_df.iloc[i,1])
            #標準化
            ST_X_test = []
            #評価データを適切な形に             
            for data in X_test:
                std = []
                for d,a,s in zip(data, AVE,STD):
                    std.append((d-a)/s)
                ST_X_test.append(std)
            pred_test = model.predict(ST_X_test)
            score = [accuracy_score(y_test, pred_test),recall_score(y_test, pred_test),precision_score(y_test, pred_test)]
            scores.append(score)
        Scores.append(np.mean(scores,axis=0))
    return(np.mean(Scores,axis=0)) 


def WA_WADP_SVM(WA,WADP, to_num, groups):
    train_df = pd.read_csv(WA,header=None,names=label)
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    #Representative 
    with open(to_num)as f:
        reader = csv.reader(f)
        To_num = [[float(v) for v in row] for row in reader]
    #生データ空間へ
    for i in range(sample_num):
        train_df.iloc[i,3:] = get_label_to_num(train_df.iloc[i,3:],To_num,framesize)
    
    models, AVEs,STDs = [],[],[]
    for IDs in crossed_ID:
        # データの分割  
        X_train = []#data
        y_train = []#target
        for i in range(sample_num):
            if train_df.iat[i,0] not in IDs:    
                X_train.append(train_df.iloc[i,3:])
                y_train.append(train_df.iloc[i,1])#ノイズなし目的変数
        AVE = []
        STD = []
        Ar_X_train = np.array(X_train)
        for i in range(framesize):#5
            feat = Ar_X_train[:,i].tolist()
            ave, std = mean_std(feat)
            AVE.append(ave)
            STD.append(std)
        AVEs.append(AVE)
        STDs.append(STD)
        #標準化
        ST_X_train = []
        #学習データの標準化
        for data in X_train:
            std = []
            for d,a,s in zip(data, AVE,STD):
                if s==0:
                    std.append(d)
                else:
                    std.append((d-a)/s)
            ST_X_train.append(std)
        #モデルの学習
        model = SVC(kernel='linear', random_state=None)
        model.fit(ST_X_train,y_train)
        models.append(model)                                                  
    Scores = [] # accuracy, recall,precision
    for wadp in WADP:
        test_df = pd.read_csv(wadp,header = None,names= label)
        #生データ空間へ
        for i in range(sample_num):
            test_df.iloc[i,3:] = get_label_to_num(test_df.iloc[i,3:],To_num,framesize)
        scores = []
        for IDs,model,AVE,STD in zip(crossed_ID,models,AVEs,STDs):
            X_test = []
            y_test = []
            for i in range(sample_num):
                if test_df.iat[i,0] in IDs:
                    X_test.append(test_df.iloc[i,3:])
                    y_test.append(test_df.iloc[i,1])
            #標準化
            ST_X_test = []
            #評価データを適切な形に             
            for data in X_test:
                std = []
                for d,a,s in zip(data, AVE,STD):
                    std.append((d-a)/s)
                ST_X_test.append(std)
            pred_test = model.predict(ST_X_test)
            score = [accuracy_score(y_test, pred_test),recall_score(y_test, pred_test),precision_score(y_test, pred_test)]
            scores.append(score)
        Scores.append(np.mean(scores,axis=0))
    return(np.mean(Scores,axis=0)) 

#---------------
#PW
#--------------  
#データセットの学習
def PW_SVM(PWdatas,groups):
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    
    Scores = [] # accuracy, recall,precision
    for PWdata in PWdatas:
        train_df = pd.read_csv(PWdata,header=None,names=label)
    
        scores = [] # accuracy, recall,precision
        model = SVC(kernel='linear', random_state=None)# 線形SVMのインスタンスを生成
        for IDs in crossed_ID:
            # データの分割  
            X_train = []#data
            y_train = []#target
            X_test = []#data
            y_test = []#target
            for i in range(sample_num):
                if train_df.iat[i,0] in IDs:
                    X_test.append(train_df.iloc[i,3:])
                    y_test.append(train_df.iloc[i,2])#ノイズ付き目的変数
                else:
                    X_train.append(train_df.iloc[i,3:])
                    y_train.append(train_df.iloc[i,1])#ノイズなし目的変数
            AVE = []
            STD = []
            Ar_X_train = np.array(X_train)
            for i in range(framesize):#5
                feat = Ar_X_train[:,i].tolist()
                ave, std = mean_std(feat)
                AVE.append(ave)#変更
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
        Scores.append(np.mean(scores,axis=0)) # accuracy,recall,precisionの平均
    return(np.mean(Scores,axis=0))


def PW_WA_SVM(PW,WA, to_num, groups):
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    
    test_df = pd.read_csv(WA,header=None,names=label) 
    #Representative 
    with open(to_num)as f:
         reader = csv.reader(f)
         To_num = [[float(v) for v in row] for row in reader]
    #生データ空間へ
    for i in range(sample_num):
        test_df.iloc[i,3:] = get_label_to_num(test_df.iloc[i,3:],To_num,framesize)
        
    Scores = [] # accuracy, recall,precision
    for pw in PW:
        train_df = pd.read_csv(pw,header=None,names=label)
    
        scores = [] # accuracy, recall,precision
        model = SVC(kernel='linear', random_state=None)# 線形SVMのインスタンスを生成
        for IDs in crossed_ID:
            # データの分割  
            X_train = []#data
            y_train = []#target
            X_test = []#data
            y_test = []#target
            for i in range(sample_num):
                if train_df.iat[i,0] in IDs:
                    X_test.append(test_df.iloc[i,3:])
                    y_test.append(test_df.iloc[i,2])#ノイズ付き目的変数
                else:
                    X_train.append(train_df.iloc[i,3:])
                    y_train.append(train_df.iloc[i,1])#ノイズなし目的変数
            AVE = []
            STD = []
            Ar_X_train = np.array(X_train)
            for i in range(framesize):#5
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
        Scores.append(np.mean(scores,axis=0)) # accuracy,recall,precisionの平均
    return(np.mean(Scores,axis=0))


def PW_WADP_SVM(PW,WADP, to_num, groups):
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    with open(to_num)as f:
        reader = csv.reader(f)
        To_num = [[float(v) for v in row] for row in reader]        
    Scores = []
    for pw,wadp in zip(PW, WADP):
        Scores.append(pw_wadp_SVM(pw, wadp, To_num, crossed_ID))
    return (np.mean(Scores,axis=0))

def pw_wadp_SVM(pw,wadp, To_num, crossed_ID):
    train_df = pd.read_csv(pw,header=None,names=label)
    test_df = pd.read_csv(wadp,header=None,names=label)
    #生データ空間へ
    for i in range(sample_num):
        test_df.iloc[i,3:] = get_label_to_num(test_df.iloc[i,3:],To_num,framesize)
    scores = [] # accuracy, recall,precision
    model = SVC(kernel='linear', random_state=None)# 線形SVMのインスタンスを生成
    for IDs in crossed_ID:
        # データの分割  
        X_train = []#data
        y_train = []#target
        X_test = []#data
        y_test = []#target
        for i in range(sample_num):
            if train_df.iat[i,0] in IDs:
                X_test.append(test_df.iloc[i,3:])
                y_test.append(test_df.iloc[i,2])#ノイズ付き目的変数
            else:
                X_train.append(train_df.iloc[i,3:])
                y_train.append(train_df.iloc[i,1])#ノイズなし目的変数
        AVE = []
        STD = []
        Ar_X_train = np.array(X_train)
        for i in range(framesize):#5
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

#---------------
#WADP
#--------------  

def WADP_WA_SVM(WADP,WA,to_num,groups):
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    #Representative 
    with open(to_num)as f:
         reader = csv.reader(f)
         To_num = [[float(v) for v in row] for row in reader]
    test_df = pd.read_csv(WA,header=None,names=label)
    #生データ空間へ
    for i in range(sample_num):
        test_df.iloc[i,3:] = get_label_to_num(test_df.iloc[i,3:],To_num,framesize)
    Scores = [] # accuracy, recall,precision
    for wadp in WADP:
        train_df = pd.read_csv(wadp,header=None,names=label)
        for i in range(sample_num):
            train_df.iloc[i,3:] = get_label_to_num(train_df.iloc[i,3:], To_num, framesize)
        scores = [] # accuracy, recall,precision
        model = SVC(kernel='linear', random_state=None)# 線形SVMのインスタンスを生成
        for IDs in crossed_ID:
            # データの分割  
            X_train = []#data
            y_train = []#target
            X_test = []#data
            y_test = []#target
            for i in range(sample_num):
                if train_df.iat[i,0] in IDs:
                    X_test.append(test_df.iloc[i,3:])
                    y_test.append(test_df.iloc[i,2])#ノイズ付き目的変数
                else:
                    X_train.append(train_df.iloc[i,3:])
                    y_train.append(train_df.iloc[i,1])#ノイズなし目的変数
            AVE = []
            STD = []
            Ar_X_train = np.array(X_train)
            for i in range(framesize):#5
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
        Scores.append(np.mean(scores,axis=0)) # accuracy,recall,precisionの平均
    return(np.mean(Scores,axis=0))

def WADP_PW_SVM(WADP, PW,to_num,groups):
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    with open(to_num)as f:
        reader = csv.reader(f)
        To_num = [[float(v) for v in row] for row in reader]        
    Scores = []
    for wadp,pw in zip(WADP, PW):
        Scores.append(wadp_pw_SVM(wadp,pw, To_num, crossed_ID))
    return (np.mean(Scores,axis=0))


def wadp_pw_SVM(wadp,pw, To_num, crossed_ID):
    train_df = pd.read_csv(wadp,header=None,names=label)
    test_df = pd.read_csv(pw,header=None,names=label)
    """
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    #Representative 
    with open(to_num)as f:
        reader = csv.reader(f)
        To_num = [[float(v) for v in row] for row in reader]
    """
    #生データ空間へ
    for i in range(sample_num):
        train_df.iloc[i,3:] = get_label_to_num(train_df.iloc[i,3:],To_num,framesize)
    scores = [] # accuracy, recall,precision
    model = SVC(kernel='linear', random_state=None)# 線形SVMのインスタンスを生成
    for IDs in crossed_ID:
        # データの分割  
        X_train = []#data
        y_train = []#target
        X_test = []#data
        y_test = []#target
        for i in range(sample_num):
            if train_df.iat[i,0] in IDs:
                X_test.append(test_df.iloc[i,3:])
                y_test.append(test_df.iloc[i,2])#ノイズ付き目的変数
            else:
                X_train.append(train_df.iloc[i,3:])
                y_train.append(train_df.iloc[i,1])#ノイズなし目的変数
        AVE = []
        STD = []
        Ar_X_train = np.array(X_train)
        for i in range(framesize):#5
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

    


def WADP_SVM(WADP,to_num,groups):
    Scores=[]
    for wadp in WADP:
        Scores.append(WA_SVM(wadp, to_num, groups))
    return(np.mean(Scores,axis=0))
        
        



    



"""
def WA_WADP_SVM(testdata,traindata, to_num, groups):
    testdf = pd.read_csv(testdata,header=None,names=label)
    traindf = pd.read_csv(traindata,header=None,names=label)
    with open(groups)as f:
        reader = csv.reader(f)
        crossed_ID = [[int(v) for v in row] for row in reader]
    #Representative 
    with open(to_num)as f:
        reader = csv.reader(f)
        To_num = [[float(v) for v in row] for row in reader]
    #生データ空間へ
    for i in range(sample_num):
        testdf.iloc[i,3:] = get_label_to_num(testdf.iloc[i,3:],To_num,framesize)
        traindf.iloc[i,3:] = get_label_to_num(traindf.iloc[i,3:],To_num,framesize)

    scores = [] # accuracy, recall,precision
    model = SVC(kernel='linear', random_state=None)# 線形SVMのインスタンスを生成
    for IDs in crossed_ID:
        # データの分割  
        X_train = []#data
        y_train = []#target
        X_test = []#data
        y_test = []#target
        for i in range(sample_num):
            if testdf.iat[i,0] in IDs:
                X_test.append(testdf.iloc[i,3:])
                y_test.append(testdf.iloc[i,2])#ノイズ付き目的変数
            else:
                X_train.append(traindf.iloc[i,3:])
                y_train.append(traindf.iloc[i,1])#ノイズなし目的変数
        AVE = []
        STD = []
        Ar_X_train = np.array(X_train)
        for i in range(framesize):#5
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
"""


                
        
            


