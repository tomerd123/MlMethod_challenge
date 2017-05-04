import sys
import time
from itertools import repeat
import io
import csv
import numpy as np
import dataSetGen as dsGen
import dA
import math
import random

def find_ngrams(input_list, n=50):
  return zip(*[input_list[i:] for i in range(n)])


def createNGramDistDic (input_list,maxN):

    allHighNGramsDic={}
    for i in range(5,maxN):
        distDic={}
        ngramsi=find_ngrams(input_list,i)
        for k in range(len(ngramsi)):
            if distDic.__contains__(ngramsi[k])==False:
                distDic[ngramsi[k]]=1
            else:
                distDic[ngramsi[k]]+=1

        allHighNGramsDic[i]=distDic
    return allHighNGramsDic

def createTrainSetWithHighNGrams (i=0,maxN=50):
    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:
        trainSegments=[]
        seg=[]
        for j, row in enumerate(file.readlines()):
            if j >= 5000:
                continue
            seg.append(row[:-1])

            if (j+1)%100==0:
                trainSegments.append(seg)
                seg=[]
    return trainSegments

def createTestSetWithHighNGrams(i=0, maxN=50):
    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:
        testSegments = []
        seg = []
        for j, row in enumerate(file.readlines()):
            if j < 5000:
                continue
            seg.append(row[:-1])

            if (j + 1) % 100 == 0:
                testSegments.append(seg)
                seg = []
    return testSegments


for seg in range(100):
    train=createTrainSetWithHighNGrams(i=5)
    test=createTestSetWithHighNGrams(i=5)
    highNTest=createNGramDistDic(test[seg],50)
    ma=0
    for k in highNTest:
       ma=max(highNTest[k][max(highNTest[k], key=lambda i: highNTest[k][i])],ma)
    print(str(seg)+","+str(ma))
print("finished")
#print(createNGramDistDic([random.uniform(1,6) for i in range(100)],50))