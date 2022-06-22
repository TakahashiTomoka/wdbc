#import Analyze
import WA_tool
import original_to_original
"""""""""""""""
Parameter Setting
"""""""""""""""
"ファイル周りの設定"
#生データ
startfile = r"..\data\wdbc.csv"
#正規化したデータ
regfile = r"..\data\wdbc_reg.csv"
#属性削減したデータ
sampledfile = r"..\data\weak_anony\wdbc_sampled.csv"
#決定した属性のデータ
attfile = r"..\data\weak_anony\decide_att.csv"
#ラベリングしたデータ
labeledfile = r"..\data\weak_anony\wdbc_label.csv"
#中央値データ
midfile = r"..\data\weak_anony\Midfile.csv"
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
sample_count = 5



def main():
    #正規化の処理
    WA_tool.reg(startfile, regfile, att_exp)
    #ランダムに属性削除
    WA_tool.random_att_del(regfile, sampledfile, att_exp, sample_count)
    #属性を決定
    WA_tool.att_decide(sampledfile, attfile, sample_count)
    #ラベリングを実行
    WA_tool.labeling(regfile, sampledfile, labeledfile, midfile)
    #SVMを実行
    print(original_to_original.CSV(labeledfile, None, groupfile, attfile))
if __name__ == "__main__":
    main()