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
import neuralNetworkBased_method as ngram
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

def normalizeGlobalDics(prob,n2,n3,n4, existTestDic,ngramStop=2):


    for k in prob:
        prob[k] /= (100.0 * 50.0)
    for k in n2:
        n2[k] /= (99.0 * 50.0)
    for k in n3:
        n3[k] /= (98.0 * 50.0)
    for k in n4:
        n4[k] /= (97.0 * 50.0)

    for j in range(100):
        if ngramStop>=1:
            for k in existTestDic[j][0]:
                existTestDic[j][0][k]=float(math.fabs(float(prob[k])-float(existTestDic[j][0][k])))
        if ngramStop>=2:
            for k in existTestDic[j][1]:
                existTestDic[j][1][k]=float(math.fabs(float(n2[k])-float(existTestDic[j][1][k])))
        if ngramStop>=3:
            for k in existTestDic[j][2]:
                existTestDic[j][2][k]=float(math.fabs(float(n3[k])-float(existTestDic[j][2][k])))
        if ngramStop>=4:
            for k in existTestDic[j][3]:
                existTestDic[j][3][k]=float(math.fabs(float(n4[k])-float(existTestDic[j][3][k])))

def getMaxAndSumAndAvgExistAndNotExist(userSegExistDic,userSegNoExistDic,j,ngramStop=2):
    sumProbDevExist = 0.0
    sumProbNotExist = 0.0
    maxProbDevExist = 0.0
    maxProbNoExist = 0.0
    avgProbExist = 0.0
    avgProbNoExist = 0.0
    sumN2DevExist = 0.0
    sumN2NotExist = 0.0
    maxN2DevExist = 0.0
    maxN2NoExist = 0.0
    avgN2Exist = 0.0
    avgN2NoExist = 0.0
    sumN3DevExist = 0.0
    sumN3NotExist = 0.0
    maxN3DevExist = 0.0
    maxN3NoExist = 0.0
    avgN3Exist = 0.0
    avgN3NoExist = 0.0
    sumN4DevExist = 0.0
    sumN4NotExist = 0.0
    maxN4DevExist = 0.0
    maxN4NoExist = 0.0
    avgN4Exist = 0.0
    avgN4NoExist = 0.0

    if ngramStop >= 1:


        for k in userSegExistDic[j][0]:
            sumProbDevExist += userSegExistDic[j][0][k]
            if userSegExistDic[j][0][k] > maxProbDevExist:
                maxProbDevExist = userSegExistDic[j][0][k]
        for k in userSegNoExistDic[j][0]:
            sumProbNotExist += userSegNoExistDic[j][0][k]
            if userSegNoExistDic[j][0][k] > maxProbNoExist:
                maxProbNoExist = userSegNoExistDic[j][0][k]
        lenProbExist=len(userSegExistDic[j][0])
        lenProbNotExist=len(userSegNoExistDic[j][0])
        if lenProbNotExist==0:
            lenProbNotExist=np.inf
        if lenProbExist==0:
            lenProbExist=np.inf
        avgProbExist=1.0-float(float(sumProbDevExist)/float(lenProbExist))
        avgProbNoExist=1.0-float(float(sumProbNotExist)/float(lenProbNotExist))
        maxProbDevExist = 1 - maxProbDevExist
        maxProbNoExist = 1 - maxProbNoExist
        sumProbDevExist = 1 - sumProbDevExist
        sumProbNotExist = 1 - sumProbNotExist
    if ngramStop>=2:


        for k in userSegExistDic[j][1]:
            sumN2DevExist += userSegExistDic[j][1][k]
            if userSegExistDic[j][1][k] > maxN2DevExist:
                maxN2DevExist = userSegExistDic[j][1][k]
        for k in userSegNoExistDic[j][1]:
            sumN2NotExist += userSegNoExistDic[j][1][k]
            if userSegNoExistDic[j][1][k] > maxN2NoExist:
                maxN2NoExist = userSegNoExistDic[j][1][k]

        lenN2Exist = len(userSegExistDic[j][0])
        lenN2NotExist = len(userSegNoExistDic[j][0])
        if lenN2NotExist == 0:
            lenN2NotExist = np.inf
        if lenN2Exist == 0:
            lenN2Exist = np.inf


        avgN2Exist = 1.0 - float(float(sumN2DevExist) / float(lenN2Exist))
        avgN2NoExist = 1.0 - float(float(sumN2NotExist) / float(lenN2NotExist))
        maxN2DevExist = 1 - maxN2DevExist
        maxN2NoExist = 1 - maxN2NoExist
        sumN2DevExist = 1 - sumN2DevExist
        sumN2NotExist = 1 - sumN2NotExist
    if ngramStop>=3:

        for k in userSegExistDic[j][2]:
            sumN3DevExist += userSegExistDic[j][2][k]
            if userSegExistDic[j][2][k] > maxN3DevExist:
                maxN3DevExist = userSegExistDic[j][2][k]
        for k in userSegNoExistDic[j][2]:
            sumN3NotExist += userSegNoExistDic[j][2][k]
            if userSegNoExistDic[j][2][k] > maxN3NoExist:
                maxN3NoExist = userSegNoExistDic[j][2][k]

        lenN3Exist = len(userSegExistDic[j][0])
        lenN3NotExist = len(userSegNoExistDic[j][0])
        if lenN3NotExist == 0:
            lenN3NotExist = np.inf
        if lenN3Exist == 0:
            lenN3Exist = np.inf

        avgN3Exist = 1.0 - float(float(sumN3DevExist) / float(lenN3Exist))
        avgN3NoExist = 1.0 - float(float(sumN3NotExist) / float(lenN3NotExist))
        maxN3DevExist = 1 - maxN3DevExist
        maxN3NoExist = 1 - maxN3NoExist
        sumN3DevExist = 1 - sumN3DevExist
        sumN3NotExist = 1 - sumN3NotExist
    if ngramStop>=4:

        for k in userSegExistDic[j][3]:
            sumN4DevExist += userSegExistDic[j][3][k]
            if userSegExistDic[j][3][k] > maxN4DevExist:
                maxN4DevExist = userSegExistDic[j][3][k]
        for k in userSegNoExistDic[j][3]:
            sumN4NotExist += userSegNoExistDic[j][3][k]
            if userSegNoExistDic[j][3][k] > maxN4NoExist:
                maxN4NoExist = userSegNoExistDic[j][3][k]

        lenN4Exist = len(userSegExistDic[j][0])
        lenN4NotExist = len(userSegNoExistDic[j][0])
        if lenN4NotExist == 0:
            lenN4NotExist = np.inf
        if lenN4Exist == 0:
            lenN4Exist = np.inf


        avgN4Exist = 1.0 - float(float(sumN4DevExist) / float(lenN4Exist))
        avgN4NoExist = 1.0 - float(float(sumN4NotExist) / float(lenN4NotExist))
        maxN4DevExist = 1 - maxN4DevExist
        maxN4NoExist = 1 - maxN4NoExist
        sumN4DevExist = 1 - sumN4DevExist
        sumN4NotExist = 1 - sumN4NotExist

    return [sumProbDevExist,sumProbNotExist,maxProbDevExist,maxProbNoExist,avgProbExist,avgProbNoExist,sumN2DevExist,sumN2NotExist,maxN2DevExist,maxN2NoExist,avgN2Exist,avgN2NoExist,sumN3DevExist,sumN3NotExist,maxN3DevExist,maxN3NoExist,avgN3Exist,avgN3NoExist,sumN4DevExist,sumN4NotExist,maxN4DevExist,maxN4NoExist,avgN4Exist,avgN4NoExist]

def calcMaxProb (userSegmentsDicProbabilityCommands,userSegmentsDic2NGram,userSegmentsDic3NGram,userSegmentsDic4NGram,j,ngramStop=2):
    maxProb = 0.0
    for k in userSegmentsDicProbabilityCommands[j]:
        if userSegmentsDicProbabilityCommands[j][k] > maxProb:
            maxProb = userSegmentsDicProbabilityCommands[j][k]
    maxProb /= 100.0

    max2NGram = 0.0
    for k in userSegmentsDic2NGram[j]:
        if userSegmentsDic2NGram[j][k] > max2NGram:
            max2NGram = userSegmentsDic2NGram[j][k]
    max2NGram /= 99.0
    max3NGram = 0.0
    for k in userSegmentsDic3NGram[j]:
        if userSegmentsDic3NGram[j][k] > max3NGram:
            max3NGram = userSegmentsDic3NGram[j][k]
    max3NGram /= 98.0
    max4NGram = 0.0
    for k in userSegmentsDic4NGram[j]:
        if userSegmentsDic4NGram[j][k] > max4NGram:
            max4NGram = userSegmentsDic4NGram[j][k]
    max4NGram /= 97.0

    return maxProb,max2NGram,max3NGram,max4NGram

def testUser (i=0,ngramStop=2):
    globalTrainProbCommandDic = dicGen.createCommandProbabilityDic(i)
    globalTrain2NGramDic = dicGen.createNGram2Dic(i)
    globalTrain3NGramDic = dicGen.createNGram3Dic(i)
    globalTrain4NGramDic = dicGen.createNGram4Dic(i)

    userSegmentsDicCommands,userSegmentsDicProbabilityCommands,userSegmentsDic2NGram,userSegmentsDic3NGram,userSegmentsDic4NGram=dicGen.testUser(i)
    userSegExistDic,userSegNoExistDic=dsGen.createTestSetForUserKnownVsUnknown(i,ngramStop)


    normalizeGlobalDics(globalTrainProbCommandDic,globalTrain2NGramDic,globalTrain3NGramDic,globalTrain4NGramDic,userSegExistDic,ngramStop)

    perSegFeatVecProb, perSegNotExistDicProb, perSegFeatVecN2, perSegNotExistDicN2, perSegFeatVecN3, perSegNotExistDicN3, perSegFeatVecN4, perSegNotExistDicN4=dsGen.createTestSetForUser(i,ngramStop)
    nnScoresProb,nnScoresN2,nnScoresN3,nnScoresN4=ngram.trainAndExecute(i,ngramStop)
    labels=[]
    for j2 in range(40):
        with io.open('E:/challenge/training_data/user2SegLabels.csv', 'rt', encoding="utf8") as file:
            file.readline()
            labList = []
            for j in range(150):
                labList.append(file.readline().split(',')[j2])
            labels.append(labList)
    print("nnScoreProb,nnScoreN2,nnScoreN3,nnScoreN4,existRatio,pos,py,lev,lcsv,ngram2,ngram3,ngram4,totalScore,knownTermsSim,maxProb,max2NGram,max3NGram,max4NGram,sumProbDevExist, sumProbNotExist, maxProbDevExist, maxProbNoExist, avgProbExist, avgProbNoExist, sumN2DevExist, sumN2NotExist, maxN2DevExist, maxN2NoExist, avgN2Exist, avgN2NoExist, sumN3DevExist, sumN3NotExist, maxN3DevExist, maxN3NoExist, avgN3Exist, avgN3NoExist, sumN4DevExist, sumN4NotExist, maxN4DevExist, maxN4NoExist, avgN4Exist, avgN4NoExist,label")
    for j in range(100):

        sumProbDevExist, sumProbNotExist, maxProbDevExist, maxProbNoExist, avgProbExist, avgProbNoExist, sumN2DevExist, sumN2NotExist, maxN2DevExist, maxN2NoExist, avgN2Exist, avgN2NoExist, sumN3DevExist, sumN3NotExist, maxN3DevExist, maxN3NoExist, avgN3Exist, avgN3NoExist, sumN4DevExist, sumN4NotExist, maxN4DevExist, maxN4NoExist, avgN4Exist, avgN4NoExist=getMaxAndSumAndAvgExistAndNotExist(userSegExistDic,userSegNoExistDic,j,ngramStop)
        maxProb,max2NGram,max3NGram,max4NGram=calcMaxProb(userSegmentsDicProbabilityCommands,userSegmentsDic2NGram,userSegmentsDic3NGram,userSegmentsDic4NGram,j,ngramStop)
        KnownTermsSim=float(calcUnknownTerms(globalTrainProbCommandDic,userSegmentsDicProbabilityCommands[j]))
        existRatio=float((float(len(userSegExistDic[j][0])+len(userSegExistDic[j][1])+len(userSegExistDic[j][2])+len(userSegExistDic[j][3]))-float(len(userSegNoExistDic[j][0])+len(userSegNoExistDic[j][1])+len(userSegNoExistDic[j][2])+len(userSegNoExistDic[j][3])))/float(len(userSegExistDic[j][0])+len(userSegExistDic[j][1])+len(userSegExistDic[j][2])+len(userSegExistDic[j][3])))
        lcs,pos,py,lev=calcPosSimilartyAndPosAndPyAndLevTestSegVsTrainsSegs(i,j)
        lev=1.0-float(lev)/100.0
        ngram2=calc2NGramJackardSim(j,i)
        ngram3=calc3NGramJackardSim(j,i)
        ngram4=calc4NGramJackardSim(j,i)
        label=labels[i+1][50+j]
        nnScoreProb=1.0-float(nnScoresProb[j])
        nnScoreN2=1.0-float(nnScoresN2[j])
        nnScoreN3=1.0-float(nnScoresN3[j])
        nnScoreN4=1.0-float(nnScoresN4[j])
        totalScore=np.array([nnScoreProb,nnScoreN2,nnScoreN3,nnScoreN4,KnownTermsSim,py,lev,existRatio,lcs,pos,maxProb,max2NGram,max3NGram,max4NGram,ngram2,ngram3,ngram4,
                             sumProbDevExist, sumProbNotExist, maxProbDevExist, maxProbNoExist, avgProbExist,
                             avgProbNoExist, sumN2DevExist, sumN2NotExist, maxN2DevExist, maxN2NoExist, avgN2Exist,
                             avgN2NoExist, sumN3DevExist, sumN3NotExist, maxN3DevExist, maxN3NoExist, avgN3Exist,
                             avgN3NoExist, sumN4DevExist, sumN4NotExist, maxN4DevExist, maxN4NoExist, avgN4Exist,
                             avgN4NoExist     ]).mean()
        #print(str(totalScore)+", nnScores is: "+str(nnScore)+", existRatio is: "+str(existRatio)+", pos is: "+str(pos)+", py is: "+str(py)+", lev is: "+str(lev)+", lcs is: "+str(lcs)+", ngram2 is: "+str(ngram2)+", ngram3 is: "+str(ngram3)+", ngram4 is: "+str(ngram4)+", totalScore is: "+ str(totalScore)+", KnownSim is: "+str(KnownTermsSim)+","+str(label))
        print(
        str(nnScoreProb)+ "," + str(nnScoreN2)+ "," + str(nnScoreN3)+ "," + str(nnScoreN4) + "," + str(existRatio) + "," + str(
            pos) + "," + str(py) + "," + str(lev) + "," + str(lcs) + "," + str(
            ngram2) + "," + str(ngram3) + "," + str(ngram4) + "," + str(
            totalScore) + "," + str(KnownTermsSim) + "," +str(maxProb)+","+str(max2NGram)+","+str(max3NGram)+","+str(max4NGram)+","+str(sumProbDevExist)+","+str(sumProbNotExist)+","+ str(maxProbDevExist)+","+ str(maxProbNoExist)+","+str( avgProbExist)+","+str( avgProbNoExist)+","+str( sumN2DevExist)+","+str( sumN2NotExist)+","+str( maxN2DevExist)+","+str( maxN2NoExist)+","+str( avgN2Exist)+","+str( avgN2NoExist)+","+str( sumN3DevExist)+","+str( sumN3NotExist)+","+str( maxN3DevExist)+","+str( maxN3NoExist)+","+str( avgN3Exist)+","+str( avgN3NoExist)+","+str( sumN4DevExist)+","+str( sumN4NotExist)+","+str( maxN4DevExist)+","+str( maxN4NoExist)+","+str( avgN4Exist)+","+str( avgN4NoExist)+"," +str(label))


"""
print ("it's 0")
testUser(0,4)

print ("it's 1")
testUser(1,4)

print ("it's 2")
testUser(2,4)

print ("it's 3")
testUser(3,4)

print ("it's 4")
testUser(4,4)

print ("it's 5")
testUser(5,4)

print ("it's 6")
testUser(6,4)

print ("it's 7")
testUser(7,4)

print ("it's 8")

testUser(8,4)
"""
print ("it's 9")

testUser(5,4)








