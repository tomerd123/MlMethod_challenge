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
from statistics import mean


def find_ngrams(input_list, n=50):
  return zip(*[input_list[i:] for i in range(n)])


def createNGramDistDic (input_list,maxN):

    allHighNGramsDic={}
    for i in range(maxN,maxN+1):
        distDic={}
        ngramsi=find_ngrams(input_list,i)
        for k in range(len(ngramsi)):
            if distDic.__contains__(ngramsi[k])==False:
                distDic[ngramsi[k]]=1
            else:
                distDic[ngramsi[k]]+=1

        allHighNGramsDic[i]=distDic
    return allHighNGramsDic
def mergeTestSegNGram (testSegNGramDic):
    testSegDic={}
    for ng in testSegNGramDic:
        for t in testSegNGramDic[ng]:
                testSegDic[t]=testSegNGramDic[ng][t]

    maxTermFreq=0
    for k in testSegDic:
        if testSegDic[k]>maxTermFreq:
            maxTermFreq=testSegDic[k]
    return testSegDic,maxTermFreq


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
def createTrainCorpus (i=0,maxN=50):
    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:
        trainTerms = {}

        ngramsTrain=createTrainSetWithHighNGrams(i,maxN)

        for seg in ngramsTrain:

            nDic=createNGramDistDic(seg,maxN)
            for ng in nDic:
                checkTerms={}
                for t in nDic[ng]:
                    if checkTerms.__contains__(t)==False:
                        checkTerms[t]=1
                    else:
                        checkTerms[t]+=1
                    if trainTerms.__contains__(t)==False:
                        trainTerms[t]=[1,1]
                    else:
                        trainTerms[t][0]=trainTerms[t][0]+nDic[ng][t]
                        if checkTerms[t]==1:
                            trainTerms[t][1]+=1
        maxFreq=0.0
        for k in trainTerms:
            maxFreq=max(maxFreq,trainTerms[k][0])

        return trainTerms,maxFreq


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


def TestUser(i=0,maxN=50):
    testSegs=createTestSetWithHighNGrams(i,maxN)
    trainCorpus,maxFreqCorp=createTrainCorpus(i,maxN)

    #calc tf-idf for corpus
    trainCorpusTfIDF={}
    for t in trainCorpus:
        if trainCorpus.__contains__(t):
            docsAppear = trainCorpus[t][1]
        else:
            docsAppear = 0.0
        trainCorpusTfIDF[t]=(0.5+0.5*float(trainCorpus[t][0])/float(maxFreqCorp))*math.log(50.0/(float(docsAppear)+0.01))

    #calc tf-idf for test-seg
    tfIDFDic={}

    print("j" + "," + "maxNotExistFreqForTerm" + "," + "sumNotExistHighNGram" + "," + "cosim"+","+"avgTestTfIDF")
    maxNotExistFreqForTermList=[]
    sumNotExistHighNGramList=[]
    cosimList=[]
    avgTestTfIDFList=[]
    for j,seg in enumerate(testSegs):
        termsNotExistDic={}
        tfIDFDic = {}
        mergedTestSegDic,maxTermFreq=mergeTestSegNGram(createNGramDistDic(seg,maxN))
        for t in mergedTestSegDic:
            if trainCorpus.__contains__(t):
                docsAppear=trainCorpus[t][1]
            else:
                docsAppear=0.0
            tfIDFDic[t]=(0.5+0.5*float(mergedTestSegDic[t])/float(maxTermFreq))*math.log(50.0/(float(docsAppear)+0.01))

        intersect=set()
        sumCorpus=0.0
        sumTestSeg=0.0
        sumIntersect=0.0
        for t1 in trainCorpus:
            sumCorpus+=trainCorpusTfIDF[t1]**2
        sumCorpus=math.sqrt(sumCorpus)
        sumNotExistHighNGram=0.0
        maxNotExistFreqForTerm=0.0
        for t2 in tfIDFDic:
            sumTestSeg+=tfIDFDic[t2]**2
            #find intersect
            if trainCorpus.__contains__(t2):
                intersect.add(t2)
            else:
                termsNotExistDic[t2]=mergedTestSegDic[t2]
                sumNotExistHighNGram+=mergedTestSegDic[t2]*len(t2)*len(t2)
                maxNotExistFreqForTerm=max(maxNotExistFreqForTerm,mergedTestSegDic[t2]*len(t2))
        sumTestSeg=math.sqrt(sumTestSeg)

        for t in intersect:
            sumIntersect+=float(trainCorpusTfIDF[t])*float(tfIDFDic[t])
        items=tfIDFDic.items()
        avgTestTfIDF=mean([x[1] for x in tfIDFDic.items()])
        cosim=float(sumIntersect)/(sumCorpus*sumTestSeg)

        print(str(j)+","+str(maxNotExistFreqForTerm)+","+str(sumNotExistHighNGram)+","+str(cosim)+","+str(avgTestTfIDF))
        maxNotExistFreqForTermList.append(maxNotExistFreqForTerm)
        sumNotExistHighNGramList.append(sumNotExistHighNGram)
        cosimList.append(cosim)
        avgTestTfIDFList.append(avgTestTfIDF)

    #min-max normalization sumNotExistHighNGramList
    minSum=min(sumNotExistHighNGramList)
    maxSum=max(sumNotExistHighNGramList)
    for s in range(len(sumNotExistHighNGramList)):
        sumNotExistHighNGramList[s]=float(float(sumNotExistHighNGramList[s]-minSum)/float(maxSum-minSum))
        sumNotExistHighNGramList[s]=sumNotExistHighNGramList[s]
    #std-mean norm for avgTestTfIDFList
    TfIDFMeanNotNormalizedList=[]
    avgTfIDFList=np.array(avgTestTfIDFList).mean()
    for a in range(len(avgTestTfIDFList)):
        TfIDFMeanNotNormalizedList.append(avgTestTfIDFList[a])
        avgTestTfIDFList[a]=math.fabs(avgTestTfIDFList[a]-avgTfIDFList)

    return maxNotExistFreqForTermList,sumNotExistHighNGramList,cosimList,avgTestTfIDFList,TfIDFMeanNotNormalizedList


for i in range (1,20):
    print ("it's i: "+str(i))
    maxNotExistFreqForTermList,sumNotExistHighNGramList,cosimList,avgTestTfIDFList,TfIDFMeanNotNormalizedList=TestUser(5,i)


print("finished high NGram")
