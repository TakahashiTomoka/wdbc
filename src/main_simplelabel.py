#import Analyze
import WA_tool
import label_LDP_tool
import PW_tool
import original_to_original
import analyze_tool
"""""""""""""""
Parameter Setting
"""""""""""""""
"ファイル周りの設定"
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

"属性の特性を定義"
#id
att_id = [0]
#ノイズ無し目的変数
att_obj = [1]
#ノイズ有り目的変数
att_noised_obj = [2]
#ノイズ有り説明変数
att_exp = list(range(3,33))

"セキュリティパラメータの設定"
#残す属性数

def main(sample_count = 5, epsilon_all = 50):
    epsilon_att = epsilon_all/(sample_count+1)
    print("sample_count:", sample_count)
    print("epsilon_all:", epsilon_all)
    print("epsilon_att:", epsilon_att)
    #正規化の処理
    WA_tool.reg(startfile, regfile, att_exp)
    #ノイズなし属性削除
    #WA_tool.random_att_del(regfile, sampledfile, att_exp, sample_count)
    #ノイズ付き属性削減
    #PW_tool.random_att_del(regfile, sampledfile, att_exp, sample_count, epsilon_att)
    #シンプルラベリング属性削減
    WA_tool.simplabel_del(regfile, simp_label_del, att_exp, sample_count)
    #属性を決定
    WA_tool.att_decide(simp_label_del, attfile, sample_count)
    #ラベリングを実行
    WA_tool.simplabel_all(regfile, simp_label, att_exp)
    #ラベリングされたデータに対しLDPを実行
    label_LDP_tool.label_LDP(simp_label, labeledldpfile, epsilon_att)
    #PWメカニズムをラベリングされていないデータに対し適用
    PW_tool.PW_all(regfile, noisedfile, epsilon_att)
    #SVMを実行
    #analyze_tool.analyze_tool(simp_label,noisedfile,labeledldpfile,midfile,groupfile,attfile)
    
if __name__ == "__main__":
    sample_count = 5
    epsilon_all = int(input("epsilon>"))
    main(sample_count ,epsilon_all)