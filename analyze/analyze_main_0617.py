# import analyze
from sklearn.metrics import classification_report
import analyze_func_0618
import glob,csv
import numpy as np
import time
import json

#Parameter Setting
sampling_num = 10 #ランダムサンプリングを行う回数
loop_num = 100 #ノイズを付ける回数
attribution = 5
classification = 5
step1 = "WA"




def ConnectParams(num,e):
    return "_" + str(num) + "(e=" +str(e) + ")"

def ConnectParams0(attribution,classification):
    return "(a=" + str(attribution) + ",c=" +str(classification) +")"

def ConnectParams1(num,attribution, classification,epsilon_all,step1):
    return str(num) + "(a=" +str(attribution) + ",c=" +str(classification) + ",e=" +str(epsilon_all) + ",step1=" + step1 +")"

def ConnectParams2(attribution, classification,step1):
    return "(a=" +str(attribution) + ",c=" +str(classification) + ",step1=" + step1 +")"

#step1 = noised専用のパラメータ
groupfile = r"/home/t-takahashi/LDP/crossd_ID.csv"
resultfile = r"/home/t-takahashi/LDP/data"+ ConnectParams0(attribution,classification) + "/result" + ConnectParams2(attribution, classification,step1) +".csv"

def main(epsilon_all, step1, train, test ):
    print("(attribution, classifivcation, step1) = ({},{},{}) ".format(attribution,classification,step1))
    print("epsilon : " + str(epsilon_all))
    print("train : " + train)
    print("test : "+ test)
    Scores = []#average, Max,min
    Times = []
    if (train, test) == ("wdbc","wdbc"):
        for num in range(sampling_num):
            #Files
            
            wdbc = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) +"/reducted_wdbc" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
            score,time = analyze_func_0618.SVM(wdbc, groupfile,attribution)
            matrix = [score,score,score]
            Scores.append([score , matrix , matrix])
            Times.append(time)
            print("Finish : " + str(num) + "/" + str(sampling_num))
    
    elif (train, test) == ("WA","WA"):
        for num in range(sampling_num):
            #Files
            midfile = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) +"/Midfile" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
            WA = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WA/wdbc_label.csv"
    
            score,time = analyze_func_0618.WA_SVM(WA, midfile, groupfile,attribution)
            matrix = [score,score,score]
            Scores.append([score , matrix , matrix])
            Times.append(time)
            print("Finish : " + str(num) + "/" + str(sampling_num))

    elif (train, test) == ("WA","PW"):
        for num in range(sampling_num):
            #Files
            midfile = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) +"/Midfile" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
            WA = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WA/wdbc_label.csv"
            PW =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/PW/*")           

            score,time = analyze_func_0618.WA_PW_SVM(WA, PW, midfile, groupfile,attribution)
            Scores.append(score)
            Times.append(time)
            print("Finish : " + str(num) + "/" + str(sampling_num))

    elif (train, test) == ("WA","WADP"):
        for num in range(sampling_num):
            #Files
            midfile = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) +"/Midfile" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
            WA = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WA/wdbc_label.csv"
            WADP =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WADP/*" ) 

            score,time = analyze_func_0618.WA_WADP_SVM(WA, WADP, midfile, groupfile,attribution)
            Scores.append(score)
            Times.append(time)
            print("Finish : " + str(num) + "/" + str(sampling_num))
#--------------------------------------
#Train = PW
#--------------------------------------
    elif (train, test) == ("PW","WA"):
        for num in range(sampling_num):
            #Files
            midfile = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) +"/Midfile" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
            WA = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WA/wdbc_label.csv"
            PW =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/PW/*")          

            score,time = analyze_func_0618.PW_WA_SVM(PW, WA, midfile, groupfile,attribution)
            Scores.append(score)
            Times.append(time)
            print("Finish : " + str(num) + "/" + str(sampling_num))

    elif (train, test) == ("PW","PW"):
        for num in range(sampling_num):
            #Files
            
          
            PW =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/PW/*")
            score,time = analyze_func_0618.PW_SVM(PW, groupfile,attribution)
            Scores.append(score)
            Times.append(time)
            print("Finish : " + str(num) + "/" + str(sampling_num))

    elif (train, test) == ("PW","WADP"):
        for num in range(sampling_num):
            #Files
            midfile = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) +"/Midfile" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
            PW =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/PW/*")          
            WADP =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WADP/*" ) 

            score,time = analyze_func_0618.PW_WADP_SVM(PW, WADP, midfile, groupfile,attribution)
            Scores.append(score)
            Times.append(time)
            print("Finish : " + str(num) + "/" + str(sampling_num))

#--------------------------------------
#Train = WADP
#--------------------------------------
    elif (train, test) == ("WADP","WA"):
        for num in range(sampling_num):
            #Files
            midfile = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) +"/Midfile" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
            WA = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WA/wdbc_label.csv"
            WADP =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WADP/*" ) 

            score,time = analyze_func_0618.WADP_WA_SVM(WADP,WA, midfile, groupfile,attribution)
            Scores.append(score)
            Times.append(time)
            print("Finish : " + str(num) + "/" + str(sampling_num))

    elif (train, test) == ("WADP","PW"):
        for num in range(sampling_num):
            #Files
            midfile = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) +"/Midfile" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
            PW =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/PW/*")           
            WADP =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WADP/*" ) 

            score,time = analyze_func_0618.WADP_PW_SVM(WADP, PW, midfile, groupfile,attribution)
            Scores.append(score)
            Times.append(time)
            print("Finish : " + str(num) + "/" + str(sampling_num))

    elif (train, test) == ("WADP","WADP"):
        for num in range(sampling_num):
            #Files
            midfile = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) +"/Midfile" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv" 
            WADP =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WADP/*" ) 

            score,time = analyze_func_0618.WADP_SVM(WADP, midfile, groupfile,attribution)
            Scores.append(score)
            Times.append(time)
            print("Finish : " + str(num) + "/" + str(sampling_num))
    else:
        matrix = [[0,0,0] for i in range(3)]
        Scores = [[[0,0,0], matrix, matrix] for i in range(10)]
        Times = [0 for i in range(10)]
        print("Wrong format")
        

    MaxScores = np.array([row[1] for row in Scores])
    minScores = np.array([row[2] for row in Scores])

    aveS = np.mean([row[0] for row in Scores],axis=0).tolist() 
    MaxS, minS = [],[]
    for i in range(3):
        Maxindex = np.argmax(MaxScores[:,i,i])
        minindex = np.argmin(minScores[:,i,i])
        MaxS.append(MaxScores[Maxindex,i,:].tolist())
        minS.append(minScores[minindex,i,:].tolist())



    # ID(attribution,classification,epsilon,step1,test,train) + time(average) + averageSCORE + maxSCORE + minSCORE
    result = [attribution,classification,epsilon,step1,train,test] + [np.mean(Times)] + aveS + sum(MaxS,[]) + sum(minS,[])
    

    
    with open(resultfile,'r',encoding="utf-8_sig") as file:
        reader = csv.reader(file)
        rowset = [[v for v in row] for row in reader]
    rowset.append(result)
    with open(resultfile, 'w',newline ="" ) as f:
        writer = csv.writer(f)
        writer.writerows(rowset)
    


    return(result)

 





for epsilon in [35,40,45,50]:
    for step1 in [step1]:
        main(epsilon,step1,"wdbc","wdbc")
        print("*****************************")
        for train in [ "WA","PW","WADP"]:
            for test in [ "WA","PW","WADP"]:
                
                main(epsilon, step1, train, test)
                print("*********************************")



