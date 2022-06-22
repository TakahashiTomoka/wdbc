# import analyze
import analyze_func
import glob,csv

#Parameter Setting

framesize  =5 #次元削減後の大きさ
sample_num = 569 #569,351
feat_num = 31 #5,30,34 #総次元数

epsilon = 50
#生データ
startfile = r"..\data\wdbc.csv"
#正規化したデータ
regfile = r"..\data\wdbc_reg.csv"
#属性削減したデータ
sampledfile = r"..\data\step1-label\wdbc_sampled.csv"
#決定した属性のデータ
attfile = r"..\data\step1-label\decide_att.csv"
#ラベリングしたデータ
#labeledfile = r"..\data\step1-label\weak_anony\wdbc_label.csv"
#中央値データ
midfile = r"..\data\step1-label\Midfile.csv"
#単純ラベリングデータ
simp_label = r"..\data\step1-label\wdbc_simplabel.csv"
#属性削減単純ラベリングデータ
simp_label_del = r"..\data\step1-label\wdbc_simplabel_del.csv"
#ラベリングデータにLDPを加えたデータ
labeledldpfile = r"..\data\step1-label\wdbc_simplabel_ldp.csv"
#非ラベリングデータにノイズを加えたデータ
noisedfile = r"..\data\step1-label\wdbc_PW_noised.csv"
#交差検定用グループデータ
groupfile = r"..\data\crossd_ID.csv"


def analyze_tool(WA,PW,WADP,to_numfile,groupfile,attfile):
    #データ加工
    """
    #analyze_func.reshape(wdbc, attfile)
    analyze_func.reshape(WA, attfile)
    analyze_func.reshape_datasets(PW, attfile)
    analyze_func.reshape_datasets(WADP, attfile)
    analyze_func.reshape_midfile(to_numfile, attfile)
    """
    analyze_func.reshape_midfile(to_numfile, attfile)
 
    
    #学習データ　= 属性削減後の生データ
    #score = analyze_func.SVM(wdbc, groupfile)
    #print(score)
    
    #学習データ　= 弱匿名化データ
    print("WA")
    score = analyze_func.WA_SVM(WA, to_numfile, groupfile)
    print(score)
    score = analyze_func.WA_PW_SVM(WA, PW, to_numfile, groupfile)
    print(score)
    score = analyze_func.WA_WADP_SVM(WA, WADP, to_numfile, groupfile)
    print(score)
    
    #学習データ　=　PW
    print("PW")
    score = analyze_func.PW_WA_SVM(PW, WA, to_numfile, groupfile)
    print(score)
    score = analyze_func.PW_SVM(PW, groupfile)
    print(score)
    score = analyze_func.PW_WADP_SVM(PW, WADP, to_numfile, groupfile)
    print(score)
    
    #学習データ　= DPWA(弱匿名化+ノイズ) 
    print("WADP")
    score = analyze_func.WADP_WA_SVM(WADP,WA, to_numfile, groupfile)
    print(score)   
    score = analyze_func.WADP_PW_SVM(WADP, PW, to_numfile, groupfile)
    print(score)
    score = analyze_func.WADP_SVM(WADP, to_numfile, groupfile)
    print(score)
    
    
if __name__ =="__main__":
    analyze_tool(simp_label,noisedfile,labeledldpfile,midfile,groupfile,attfile)