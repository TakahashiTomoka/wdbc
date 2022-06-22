import numpy as np
from statistics import mean, median,variance,stdev
import scipy.stats
import statistics
import pandas as pd
import array
import random
import os
import glob
import time
import csv

framesize  =5 #次元削減後の大きさ
sample_num = 569 #569,351
feat_num = 31 #5,30,34 #総次元数

def reshape(labeledfile, attfile):
    #属性読み込み
    with open(attfile) as file:
        reader = csv.reader(file)
        att = [row for row in reader]
    att = [int(v) for v in att[0]]    
    #選択属性のみのファイルに書き直し
    with open(labeledfile) as file:
        reader = csv.reader(file)
        rowset = [[row[i] for i in att] for row in reader]
    with open(labeledfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rowset)

def reshape2(inputfile,createfile, attfile):
    #属性読み込み
    with open(attfile) as file:
        reader = csv.reader(file)
        att = [row for row in reader]
    att = [int(v) for v in att[0]]    
    #選択属性のみのファイルに書き直し
    with open(inputfile) as file:
        reader = csv.reader(file)
        rowset = [[row[i] for i in att] for row in reader]
    with open(createfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rowset)



def reshape_datasets(folder,attfile):
    for file in folder:
        reshape(file,attfile)
        
#-------------------


#中間値データのreshape        
def reshape_midfile(midfile, attfile):
    #属性読み込み
    with open(attfile) as file:
        reader = csv.reader(file)
        att = [row for row in reader]
    att = [int(v)-3 for v in att[0]]    
    #選択属性のみのファイルに書き直し
    with open(midfile) as file:
        reader = csv.reader(file)
        rowset = [[v for v in row] for row in reader]
    midset = []
    for i in att[3:]:
        midset.append(rowset[i])
    with open(midfile, 'w',newline="") as file:
        writer = csv.writer(file)
        writer.writerows(midset)
