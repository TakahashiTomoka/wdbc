from hashlib import new
import LDP
import Analyze
import csv
import Split
import numpy as np

def ConnectParams(k,m,f,q,p):
    return "f=" + str(f) + ",q=" + str(q) + ",p=" + str(p)

def loop_main(loopNum):
    for loop in range(loopNum):
        """""""""""""""
        Parameter Setting
        """""""""""""""
        # 注意: Midfileはmain.pyでしか生成されません．同じパラメータで事前にmain.pyを実行しておくことをお勧めします．
        
        # k,m,saltsはBFに使うパラメータなので，ラベリングをする場合は無視してOKです．
        k = 10
        m = 10000 
        salts = [str(i) for i in range(k)]
        
        # f: a probability for randamization (keep almost the original BF)
        # q: randomization of 1s in BF (q=1 -> keep 1, q=0 -> reversed to 0)
        # p: randomization of 1s in BF (p=1 -> reversed to 1, p=0 -> keep 0)
        
        f = 0.9004 #[0.28(ε1),0.55(ε2),0.65(ε3)]
                 #[0.5379(εm=10),0.6201(εm=8),0.7087(εm=6),0.8027(εm=4),0.9004(εm=2)]
        # 基本的にはq=1,p=0で固定．fはCalcEp.pyでεから逆算して求めると良いです．
        q = 1
        p = 0
        # 説明変数以外のインデックス
        skipindex = [0,1]
        #生データのパス
        openfile = r"C:\Users\yoshi\OneDrive\Documents\Lecture\Prosec\data\wdbc(coef).csv" 
        # 生成した汎化データを保存するパス
        labeledfile = r"C:\Users\yoshi\OneDrive\Documents\Lecture\Prosec\data\Labeledfile(coef).csv" 
        # 生成したLDPデータを保存するパス
        createfile = r"C:\Users\yoshi\OneDrive\Documents\Lecture\Prosec\largeData\εm=2\LDPfile(" + ConnectParams(k,m,f,q,p) + ")_" + str(loop) + ".csv" 
        """""""""""""""
        Parameter Setting
        """""""""""""""
        
        # 要素数分割ではなく区間長分割も実装済みですが，ちょっと操作が難しいので基本的に以下は触らなくいて良いです．

        #Step1: split all items in csv according to the range of each attribute.
            #case1: split according to the same length of the entire range.
        #splitset = Split.getSplitTupleSetForAttributes(openfile,skipindex,5,100)
            #case2: split according to the same number of attributes.
        splitset, midSet = Split.getSplitTupleSetForAttributes2(openfile,skipindex,5)
        Split.makeLabeledFile(openfile,labeledfile,splitset,skipindex)
        splitnum = [len(splitTuple) for splitTuple in splitset]

        #Step1: input all items in csv onto a LDP protocol
        #case1->openfile
        #case2->labeledfile
        with open(labeledfile) as file:
            reader = csv.reader(file)
            rowset = [row for row in reader]
        LDPset = []
        for row in rowset:
            newrow = []
            for i in range(len(row)):
                if i in skipindex:
                    newrow.append(row[i])
                    continue    
                """
                #case1: construction BF
                ItemSet = LDP.BloomFilter(k,m,salts)
                ItemSet.setBF(str(row[i])) 
                """
                
                #case2: construction ItemBox
                ItemSet = LDP.ItemBox(splitnum[i])
                ItemSet.setItem(int(row[i])) 
                
                
                S = LDP.LDP("ItemBox",ItemSet,f,q,p)
                #convert a LDP output (array) to one string
                newitem = "".join(list(map(str,S)))
                newrow.append(float(newitem))
            LDPset.append(newrow)
        
        #Step2: Save LDP's outputs as a csv file
        with open(createfile, 'w',newline="") as file:
            writer = csv.writer(file)
            writer.writerows(LDPset)
        
        print("finish"+ str(loop+1) + "/" + str(loopNum))
        
loop_main(1000)