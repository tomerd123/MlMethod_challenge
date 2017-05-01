import sys
import time
from itertools import repeat
import io
import csv
import numpy as np
import dataSetGen as dsGen
import dictionaryGen as dicGen
import dA
import math
from sets import Set
import ngramBased_method as ngram
from difflib import SequenceMatcher

def calculatePosSimilarity (s1,s2):
    count=0
    for t in range(len(s1)):
        if s1[t]==s2[t]:
            count+=1
    return float(float(count)/float(len(s1)))

def lcs_length(a, b):
    table = [[0] * (len(b) + 1) for _ in xrange(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]

def nLCS (a,b,lcsLength):
    return float(lcsLength/(math.sqrt(len(a)*len(b))))

def calcPosSimilartyAndPosAndPyAndLevTestSegVsTrainsSegs (i=0,testSeg=0):
    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:
        lines = file.readlines()

        trainSegs=[]
        testSegs=[]
        trainSeg = []
        for j,row in enumerate(lines):
            if j%100==0 and j!=0 and j<=5000:
                trainSegs.append(trainSeg)
                trainSeg=[]

            if j==5000+testSeg*100:
                trainSeg = []
            if j!=14999 and j==5000+testSeg*100+100:
                testSegs.append(trainSeg)
                break
            if j!=14999:
                trainSeg.append(row[:-1])

            if  j==14999:
                trainSeg.append(row[:-1])
                testSegs.append(trainSeg)


    simLCSScores=[]
    simPosScores=[]
    simPythonScores=[]
    simLevScores=[]
    for s in trainSegs:
        lcsL=lcs_length(s,testSegs[0])
        lcs=nLCS(s,testSegs[0],lcsL)
        pos=calculatePosSimilarity(s,testSegs[0])
        sm = SequenceMatcher(None, s, testSegs[0])
        pythonMatch = sm.ratio()
        lev=levenshtein(s,testSegs[0])
        simPythonScores.append(pythonMatch)
        simLCSScores.append(lcs)
        simPosScores.append(pos)
        simLevScores.append(lev)
    return np.array(simLCSScores).mean(),np.array(simPosScores).mean(),np.array(simPythonScores).mean(),np.array(simLevScores).mean()


def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def calc2NGramJackardSim (s1,i=0):

    ngram2TrainDic=dicGen.createPerSegmentNGram2Dic(i)
    userSegmentsDicCommands, userSegmentsDicProbabilityCommands, userSegmentsDic2NGram, userSegmentsDic3NGram, userSegmentsDic4NGram = dicGen.testUser(i)

    disjunction=set()
    union=set()

    simScores=[]
    for seg in range(len(ngram2TrainDic)): #for each train segment
        union=set()
        disjunction=set()
        for k in ngram2TrainDic[seg]:
            for k2 in userSegmentsDic2NGram[int(s1)]:
                if k==k2:
                    disjunction.add(k)
                    union.add(k)
                else:
                    union.add(k)
                    union.add(k2)
        score=float(float(len(disjunction))/float(len(union)))
        simScores.append(score)
    return np.array(simScores).mean()

def calc3NGramJackardSim (s1,i=0):

    ngram3TrainDic=dicGen.createPerSegmentNGram3Dic(i)
    userSegmentsDicCommands, userSegmentsDicProbabilityCommands, userSegmentsDic2NGram, userSegmentsDic3NGram, userSegmentsDic4NGram = dicGen.testUser(i)

    disjunction=set()
    union=set()

    simScores=[]
    for seg in range(len(ngram3TrainDic)): #for each train segment
        union=set()
        disjunction=set()
        for k in ngram3TrainDic[seg]:
            for k2 in userSegmentsDic3NGram[int(s1)]:
                if k==k2:
                    disjunction.add(k)
                    union.add(k)
                else:
                    union.add(k)
                    union.add(k2)
        score=float(float(len(disjunction))/float(len(union)))
        simScores.append(score)
    return np.array(simScores).mean()

def calc4NGramJackardSim (s1,i=0):

    ngram4TrainDic=dicGen.createPerSegmentNGram4Dic(i)
    userSegmentsDicCommands, userSegmentsDicProbabilityCommands, userSegmentsDic2NGram, userSegmentsDic3NGram, userSegmentsDic4NGram = dicGen.testUser(i)

    disjunction=set()
    union=set()

    simScores=[]
    for seg in range(len(ngram4TrainDic)): #for each train segment
        union=set()
        disjunction=set()
        for k in ngram4TrainDic[seg]:
            for k2 in userSegmentsDic4NGram[int(s1)]:
                if k==k2:
                    disjunction.add(k)
                    union.add(k)
                else:
                    union.add(k)
                    union.add(k2)
        score=float(float(len(disjunction))/float(len(union)))
        simScores.append(score)
    return np.array(simScores).mean()

def calcUnknownTerms (globDic,segDic):
    union=set()
    intersec=set()
    for k in globDic:
        union.add(k)
        if segDic.__contains__(k):
            intersec.add(k)
    score = float(float(len(intersec)) / float(len(union)))
    return score


def testUser (i=0):
    globalTrainProbCommandDic = dicGen.createCommandProbabilityDic(i)
    userSegmentsDicCommands,userSegmentsDicProbabilityCommands,userSegmentsDic2NGram,userSegmentsDic3NGram,userSegmentsDic4NGram=dicGen.testUser(i)


    perSegFeatVec,perSegNotExistDic=dsGen.createTestSetForUser(i)
    nnScores=ngram.trainAndExecute(i)
    labels=[]
    for j2 in range(40):
        with io.open('E:/challenge/training_data/user2SegLabels.csv', 'rt', encoding="utf8") as file:
            file.readline()
            labList = []
            for j in range(150):
                labList.append(file.readline().split(',')[j2])
            labels.append(labList)
    print("nnScore,existRatio,pos,py,lev,lcsv,ngram2,ngram3,ngram4,totalScore,knownTermsSim,maxProb,max2NGram,max3NGram,max4NGram,label")
    for j in range(100):

        maxProb=0.0
        for k in userSegmentsDicProbabilityCommands[j]:
            if userSegmentsDicProbabilityCommands[j][k]>maxProb:
                maxProb=userSegmentsDicProbabilityCommands[j][k]
        maxProb/=100.0

        max2NGram=0.0
        for k in userSegmentsDic2NGram[j]:
            if userSegmentsDic2NGram[j][k]>max2NGram:
                max2NGram=userSegmentsDic2NGram[j][k]
        max2NGram/=99.0
        max3NGram = 0.0
        for k in userSegmentsDic3NGram[j]:
            if userSegmentsDic3NGram[j][k] > max3NGram:
                max3NGram = userSegmentsDic3NGram[j][k]
        max3NGram/=98.0
        max4NGram = 0.0
        for k in userSegmentsDic4NGram[j]:
            if userSegmentsDic4NGram[j][k] > max4NGram:
                max4NGram = userSegmentsDic4NGram[j][k]
        max4NGram/=97.0
        KnownTermsSim=float(calcUnknownTerms(globalTrainProbCommandDic,userSegmentsDicProbabilityCommands[j]))
        existRatio=float((float(len(perSegFeatVec[j]))-float(len(perSegNotExistDic[j])))/float(len(perSegFeatVec[j])))
        lcs,pos,py,lev=calcPosSimilartyAndPosAndPyAndLevTestSegVsTrainsSegs(i,j)
        lev=1.0-float(lev)/100.0
        ngram2=calc2NGramJackardSim(j,i)
        ngram3=calc3NGramJackardSim(j,i)
        ngram4=calc4NGramJackardSim(j,i)
        label=labels[i+1][50+j]
        nnScore=1.0-float(nnScores[j][0])
        totalScore=np.array([nnScore,KnownTermsSim,py,lev,existRatio,lcs,pos,maxProb,max2NGram,max3NGram,max4NGram,ngram2,ngram3,ngram4]).mean()
        #print(str(totalScore)+", nnScores is: "+str(nnScore)+", existRatio is: "+str(existRatio)+", pos is: "+str(pos)+", py is: "+str(py)+", lev is: "+str(lev)+", lcs is: "+str(lcs)+", ngram2 is: "+str(ngram2)+", ngram3 is: "+str(ngram3)+", ngram4 is: "+str(ngram4)+", totalScore is: "+ str(totalScore)+", KnownSim is: "+str(KnownTermsSim)+","+str(label))
        print(
        str(nnScore) + "," + str(existRatio) + "," + str(
            pos) + "," + str(py) + "," + str(lev) + "," + str(lcs) + "," + str(
            ngram2) + "," + str(ngram3) + "," + str(ngram4) + "," + str(
            totalScore) + "," + str(KnownTermsSim) + "," +str(maxProb)+","+str(max2NGram)+","+str(max3NGram)+","+str(max4NGram)+","+ str(label))



testUser(7)


