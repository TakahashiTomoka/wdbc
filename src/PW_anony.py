#import Analyze
import PW_tool
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
sampledfile = r"..\data\PW_anony\wdbc_sampled.csv"
#決定した属性のデータ
attfile = r"..\data\PW_anony\decide_att.csv"
#ノイズ付与したデータ
noisedfile = r"..\data\PW_anony\wdbc_noised.csv"
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
#全体でのepsilon
epsilon_all = 5
#各属性のepsilon
epsilon_att = epsilon_all/(sample_count + 1)



def main():
    #正規化の処理
    PW_tool.reg(startfile, regfile, att_exp)
    #ランダムに属性削除及び残ったデータに対してPWメカニズムの適用
    PW_tool.random_att_del(regfile, sampledfile, att_exp, sample_count, epsilon_att)
    #属性を決定
    PW_tool.att_decide(sampledfile, attfile, sample_count)
    #PWメカニズムを加え送信
    PW_tool.PW_all(regfile, noisedfile, epsilon_att)
    #SVMを実行
    print(original_to_original.CSV(noisedfile, None, groupfile, attfile))
if __name__ == "__main__":
    main()