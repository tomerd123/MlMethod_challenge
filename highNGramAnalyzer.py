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
        trainCorpusTfIDF[t]=(0.5+0.5*float(trainCorpus[t][0])/(float(maxFreqCorp)+1))*math.log(50.0/(float(docsAppear)+1))

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
            tfIDFDic[t]=(0.5+0.5*float(mergedTestSegDic[t])/(float(maxTermFreq)+1))*math.log(50.0/(float(docsAppear)+1))

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

def findMaxFreqInSeg (seg):
    m=0.0
    for k in seg:
        if seg[k]>m:
            m=seg[k]
    return m

def calcCosimTestVsAllTrains(trainSegsTfIDF,testSeg):
    sumTrainSegsDic=[]

    sumTrain=0.0
    for seg in range(len(trainSegsTfIDF)):
        for t in trainSegsTfIDF[seg]:
            sumTrain+=trainSegsTfIDF[seg][t]**2
        sumTrain=math.sqrt(sumTrain)
        sumTrainSegsDic.append(sumTrain)

    sumTest=0.0
    for t in testSeg:
        sumTest+=testSeg[t]**2
    sumTest=math.sqrt(sumTest)

    sumSim=0.0
    countTrain=50.0
    sumNumerator=0.0


    for trainSeg in range(len(trainSegsTfIDF)):
        sumNumerator=0.0
        for t in trainSegsTfIDF[trainSeg]:
            if testSeg.__contains__(t):
                sumNumerator+=trainSegsTfIDF[trainSeg][t]*testSeg[t]
        sim=sumNumerator/(sumTest*sumTrainSegsDic[trainSeg])


        sumSim+=sim

    avgSim=sumSim/countTrain



    return avgSim



def TestUserWithSeg2SegTfIdf (i=0, maxN=50):

    testSegs = createTestSetWithHighNGrams(i, maxN)
    trainCorpus, maxFreqCorp = createTrainCorpus(i, maxN)
    trainSegs=createTrainSetWithHighNGrams(i,maxN)

    testSegsTfIDF = []
    #calc tfIdf for test-segs
    for testSeg in testSegs:
        testSegDist=createNGramDistDic(testSeg,maxN)
        distDic,maxFr=mergeTestSegNGram(testSegDist)
        tfIdfSeg={}
        for t in distDic:
            if trainCorpus.__contains__(t):
                docsAppear = trainCorpus[t][1]
            else:
                docsAppear = 0.0
            distDicMerged,maxF=mergeTestSegNGram(testSegDist)
            tfIdfSeg[t] = (0.5 + 0.5 * (float(distDicMerged[t]) / (float(findMaxFreqInSeg(distDicMerged))+1))) * math.log(50.0 / (float(docsAppear) + 1))
        testSegsTfIDF.append(tfIdfSeg)


    trainSegsTfIDF = []
    # calc tfIdf for train-segs
    for trainSeg in trainSegs:
        trainSegDist = createNGramDistDic(trainSeg, maxN)
        trainSegDist,maxFre=mergeTestSegNGram(trainSegDist)
        tfIdfSeg = {}
        for t in trainSegDist:
            if trainCorpus.__contains__(t):
                docsAppear = trainCorpus[t][1]
            else:
                docsAppear = 0.0
            tfIdfSeg[t] = (0.5 + 0.5 * (float(trainSegDist[t]) / float(findMaxFreqInSeg(trainSegDist)+1))) * math.log(
            50.0 / (float(docsAppear) + 1))
        trainSegsTfIDF.append(tfIdfSeg)

    #calc sim foreach test Vs all train segs

    cosimTestSegsList=[]




    for testSeg in testSegsTfIDF:
            sim=calcCosimTestVsAllTrains(trainSegsTfIDF,testSeg)
            cosimTestSegsList.append(sim)

    return cosimTestSegsList



i=5

print ("it's i: "+str(i))
sim1=TestUserWithSeg2SegTfIdf(i,1)
sim2=TestUserWithSeg2SegTfIdf(i,2)
sim3=TestUserWithSeg2SegTfIdf(i,3)
sim4=TestUserWithSeg2SegTfIdf(i,4)
sim5=TestUserWithSeg2SegTfIdf(i,5)
sim6=TestUserWithSeg2SegTfIdf(i,6)
sim7=TestUserWithSeg2SegTfIdf(i,7)
sim8=TestUserWithSeg2SegTfIdf(i,8)
for s in range(len(sim1)):
    avg=(sim1[s]+sim2[s]+sim3[s]+sim4[s]+sim5[s]+sim6[s]+sim7[s]+sim8[s])/8.0
    print(str(s)+","+str(sim1[s])+","+str(sim2[s])+","+str(sim3[s])+","+str(sim4[s])+","+str(sim5[s])+","+str(sim6[s])+","+str(sim7[s])+","+str(sim8[s])+",      "+str(avg))



print("finished high NGram")