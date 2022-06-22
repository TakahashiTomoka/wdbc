# import analyze
import analyze_func
import glob,csv

#Parameter Setting

framesize  =5 #次元削減後の大きさ
sample_num = 569 #569,351
feat_num = 31 #5,30,34 #総次元数

epsilon = 50
step1 = "step1-noised" # "step1-noised" or "step1-unnoised"

wdbc = r"../data/e="  + str(epsilon) + "/" + step1 + "/reducted_wdbc.csv"
WA = r"../data/e=" + str(epsilon) + "/" + step1 + "/weak_anony/wdbc_label.csv"
PW = glob.glob(r"../data/e=" + str(epsilon) + "/" + step1 + "/PW_anony/*")
WADP = glob.glob(r"../data/e=" + str(epsilon) + "/" + step1 + "/label_LDP_anony/*")
to_numfile = r"../data/e=" + str(epsilon) + "/" + step1 + "/Midfile.csv"
groupfile = r"../data/crossd_ID.csv"
attfile = r"../data/e=" + str(epsilon) + "/" + step1 + "/decide_att.csv"




def main():
    #データ加工
    """
    analyze_func.reshape(wdbc, attfile)
    analyze_func.reshape(WA, attfile)
    analyze_func.reshape_datasets(PW, attfile)
    analyze_func.reshape_datasets(WADP, attfile)
    analyze_func.reshape_midfile(to_numfile, attfile)
    """
 
    
    #学習データ　= 属性削減後の生データ
    score = analyze_func.SVM(wdbc, groupfile)
    print(score)
    
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
    
    
if __name__ == "__main__":
    main()  