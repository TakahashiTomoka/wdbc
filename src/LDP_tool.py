import numpy as np
import random
import math


#PWメカニズム
def Piecewise(t, epsilon):
    C = (math.exp(epsilon/2)+1)/(math.exp(epsilon/2)-1)
    l = ((C+1)/2)*t - (C-1)/2
    r = l + C -1
    
    x = random.random()
    
    if x < math.exp(epsilon/2)/(math.exp(epsilon/2)+1):
        return random.uniform(l, r)
    else:
        rate = abs(l+C) /(abs(l+C)+abs(C-r))
        x = random.random()
        if x < rate:
            return random.uniform(-C, l)
        else:
            return random.uniform(r, C)

#一様ランダム
def random_uniform(t, epsilon0):
    C = (math.exp(epsilon0/2)+1)/(math.exp(epsilon0/2)-1)
    return random.uniform(-C, C)

#1bitに対するノイズ
def one_bit_LDP(t, epsilon0):
    C = (math.exp(epsilon0))/(1+math.exp(epsilon0))
    x = random.random()
    
    if x <= C:
        if t == 0:
            return 0
        else:
            return 1
    else:
        if t == 0:
            return 1
        else:
            return 0
        
        
#ラベル化されたデータに対するノイズ
def label(t, epsilon_att, label_num):
    t = int(t)
    fail_label = list(range(label_num))
    fail_label.pop(t)
    #print(t,fail_label)
    ans_label = [t]+fail_label
    C = (math.exp(epsilon_att))/(label_num-1+math.exp(epsilon_att))
    D = 1/(label_num-1+math.exp(epsilon_att))
    x = random.random()    
    for i in range(5):#細かく言うと、 label_numになる。しかし、今回はlabel_num(=classification)=3,4,5なので、問題なし。
        if x <= C:
            return ans_label[i]
        else:
            C = C + D
    return label_num - 1