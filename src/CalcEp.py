from numpy import log as ln
import math

# e1: Basic One-time RAPPORのε
# e2: Basic RAPPORのε(多分もう使わないかな...)
f= 0.3
p = 0
q = 0.75
h = 1
qt = 0.5*f*(p+q)+(1-f)*q
pt = 0.5*f*(p+q)+(1-f)*p

e1 = 2*h*ln((1-0.5*f)/(0.5*f))
e2 = h*ln((qt*(1-pt))/(pt*(1-qt)))

#print(e1)
#print(e2)

# ここからはεからBasic One-time RAPPORのパラメータfを逆算する．(総当たり)
E = 2/5

for i in range(10**4):
    f= (i+1)/(10**4)
    p = 0
    q = 1
    qt = 0.5*f*(p+q)+(1-f)*q
    pt = 0.5*f*(p+q)+(1-f)*p
    e = h*ln((qt*(1-pt))/(pt*(1-qt)))
    if e<E:
        print("f:"+str(f))
        print(e)
        break

    