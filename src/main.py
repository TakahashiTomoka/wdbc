#import Analyze
import WA_tool
import label_LDP_tool
import PW_tool
import reshape
import glob
"""""""""""""""
Parameter Setting
"""""""""""""""
"ファイル周りの設定"


step1 = "random" #unnoised,PW, WA,WADP
#生データ
startfile = r"/home/t-takahashi/LDP/wdbc.csv"
#正規化したデータ
regfile = r"/home/t-takahashi/LDP/wdbc_reg.csv"


def ConnectParams(num,e):
    return "_" + str(num) + "(e=" +str(e) + ")"

def ConnectParams0(attribution,classification):
    return "(a=" + str(attribution) + ",c=" +str(classification) +")"

def ConnectParams1(num,attribution, classification,epsilon_all,step1):
    return str(num) + "(a=" +str(attribution) + ",c=" +str(classification) + ",e=" +str(epsilon_all) + ",step1=" + step1 +")"





"属性の特性を定義"
#id
att_id = [0]
#ノイズ無し目的変数
att_obj = [1]
#ノイズ有り目的変数
att_noised_obj = [2]
#ノイズ有り説明変数
att_exp = list(range(3,33))


def main(attribution = 5, epsilon_all = 50, classification=5, sampling_num = 10, loop_num = 10):
    print("attribution:", attribution)
    print("classification :", classification)
    print("epsilon_all:", epsilon_all)
    print("Step1 :",step1)
    
    #正規化の処理
    WA_tool.reg(startfile, regfile, att_exp)
    for num in range(sampling_num):
        #files 
        sampledfile = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1+ "/" + str(num) + "/wdbc_sampled" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
        attfile = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1+ "/" + str(num) + "/decide_att" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
        midfile = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) +"/Midfile" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
        wdbc = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) +"/reducted_wdbc" + ConnectParams1(num,attribution, classification,epsilon_all,step1)+".csv"
        wa = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WA/wdbc_label.csv"
        #Step1**********************
        #ノイズなし属性削除(Step1=unnoised) + 属性を決定　+　ラベリング
        if step1 == "random":
            epsilon_att = epsilon_all/((attribution+1))
            WA_tool.random_att_del(regfile, sampledfile, att_exp, attribution)
            WA_tool.random_att_decide(sampledfile,attfile, attribution)
            WA_tool.simplabel_all(regfile, wa, midfile, att_exp, classification)

        
        elif step1 == "PW":
            #ノイズ付き属性削減(Step1=PW)　+ 属性を決定　+　ラベリング
            epsilon_att = epsilon_all/((attribution+1)*2)
            PW_tool.random_att_del(regfile, sampledfile, att_exp, attribution, epsilon_att)
            WA_tool.att_decide(sampledfile, attfile, attribution)
            WA_tool.simplabel_all(regfile, wa, midfile, att_exp, classification)
        
        elif step1 =="WA":
            #シンプルラベリング属性削減(Step1 = WA)　+ 属性を決定　+　ラベリング
            epsilon_att = epsilon_all/((attribution+1))
            WA_tool.simplabel_del(regfile, sampledfile, midfile, att_exp, attribution, classification)
            WA_tool.att_decide(sampledfile, attfile, attribution)
            WA_tool.simplabel_all(regfile, wa, midfile, att_exp, classification)
           
        
        elif step1 =="WADP":
        #シンプルラベリング属性削減 + WADP(Step1 = WADP)　+ 属性を決定
            epsilon_att = epsilon_all/((attribution+1)*2)
            WA_tool.simplabel_del(regfile, simp_label, att_exp, attribution)
            WA_tool.att_decide(sampledfile, attfile, attribution)
            #ラベリング方法の確認
        else:
            print("error")
        
        if num == 0:
            print("epsilon_att : {}".format(epsilon_att))


        
        #*****************
        for loop in range(loop_num):
            #files
            wadp = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WADP/wadp" + ConnectParams1(num,attribution, classification,epsilon_all,step1) +"_"+ str(loop) +".csv"
            pw = r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/PW/pw" + ConnectParams1(num,attribution, classification,epsilon_all,step1)  +"_"+ str(loop) +".csv"
            
            
            #汎化されたデータに対しLDPを実行(wadp)
            label_LDP_tool.label_LDP(wa, wadp, epsilon_att, classification)
            #PWメカニズムを連続データに対し適用(pw)
            PW_tool.PW_all(regfile, pw, epsilon_att)
        
        print("Finish : " + str(num+1) + "/" + str(sampling_num))
        WADP =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/WADP/*" ) 
        PW =glob.glob(r"/home/t-takahashi/LDP/data" + ConnectParams0(attribution, classification) +"/e=" + str(epsilon_all) + "/step1=" + step1 + "/" + str(num) + "/PW/*")             
        
        reshape.reshape2(regfile,wdbc, attfile)
        reshape.reshape(wa, attfile)
        reshape.reshape_datasets(PW, attfile)
        reshape.reshape_datasets(WADP, attfile)
        reshape.reshape_midfile(midfile, attfile)
    
"""" 
if __name__ == "__main__":
    attribution =5 #int(input("a (attribution)>"))
    epsilon_all = int(input("epsilon>"))
    classification =3 #int(input("c (classification)>"))
    sampling_num = 10
    loop_num = 100
    main(attribution ,epsilon_all, classification,sampling_num,loop_num)
"""
if __name__ == "__main__":
    sampling_num = 10
    loop_num = 100
    
    for attribution in [5]:
        for classification in [5]:
            for epsilon_all in [25,30,35,40,45,50]:
                main(attribution ,epsilon_all, classification,sampling_num,loop_num)