import sys
import time
from itertools import repeat
import io
import csv
import numpy as np
import dictionaryGen as dicGen


def createTrainSetForUser(i=0,ngramStop=2):
    globalTrainCommandPosDic, globalTrain2NGramDic, globalTrain3NGramDic, globalTrain4NGramDic, globalTrainProbCommandDic,perSegTrainCommandPosDic, perSegTrain2NGramDic, perSegTrain3NGramDic, perSegTrain4NGramDic, perSegTrainProbCommandDic,userSegmentsDicCommands, userSegmentsDicProbabilityCommands, userSegmentsDic2NGram, userSegmentsDic3NGram, userSegmentsDic4NGram=dicGen.getDics(i)

    globalFeatVecProb=[]
    globalFeatVecN2=[]
    globalFeatVecN3=[]
    globalFeatVecN4=[]

    """for k in globalTrainCommandPosDic:
        globalFeatVec.append(float(float(globalTrainCommandPosDic[k])/50.0))"""
    if ngramStop>=1:
        for k in globalTrainProbCommandDic:
            globalFeatVecProb.append(float(float(globalTrainProbCommandDic[k])/5000.0))
    if ngramStop>=2:
        for k in globalTrain2NGramDic:
            globalFeatVecN2.append(float(float(globalTrain2NGramDic[k])/(99.0*50.0)))
    if ngramStop>=3:
        for k in globalTrain3NGramDic:
            globalFeatVecN3.append(float(float(globalTrain3NGramDic[k]) / (98.0 * 50.0)))
    if ngramStop>=4:
        for k in globalTrain4NGramDic:
            globalFeatVecN4.append(float(float(globalTrain4NGramDic[k]) / (97.0 * 50.0)))

    perSegFeatVecProb=[]
    segFeatVecProb=[]
    perSegFeatVecN2 = []
    segFeatVecN2 = []
    perSegFeatVecN3= []
    segFeatVecN3 = []
    perSegFeatVecN4 = []
    segFeatVecN4 = []
    for j in range(50):
        """for k in globalTrainCommandPosDic:
            if perSegTrainCommandPosDic[j].__contains__(k):
                segFeatVec.append(float(float(perSegTrainCommandPosDic[j][k]) / (float(globalTrainCommandPosDic[k]))))
            else:
                segFeatVec.append(float(0.0))"""
        if ngramStop>=1:
            for k in globalTrainProbCommandDic:
                if perSegTrainProbCommandDic[j].__contains__(k):
                    segFeatVecProb.append(
                        float(float(perSegTrainProbCommandDic[j][k]) / ( float(globalTrainProbCommandDic[k]))))
                else:
                    segFeatVecProb.append(float(0.0))
        if ngramStop>=2:
            for k in globalTrain2NGramDic:
                if perSegTrain2NGramDic[j].__contains__(k):
                    segFeatVecN2.append(
                        float(float(perSegTrain2NGramDic[j][k]) / (float( globalTrain2NGramDic[k]))))
                else:
                    segFeatVecN2.append(float(0.0))
        if ngramStop>=3:
            for k in globalTrain3NGramDic:
                if perSegTrain3NGramDic[j].__contains__(k):
                    segFeatVecN3.append(
                        float(float(perSegTrain3NGramDic[j][k]) / (float(globalTrain3NGramDic[k]))))
                else:
                    segFeatVecN3.append(float(0.0))
        if ngramStop>=4:
            for k in globalTrain4NGramDic:
                if perSegTrain4NGramDic[j].__contains__(k):
                    segFeatVecN4.append(
                        float(float(perSegTrain4NGramDic[j][k]) / (float(globalTrain4NGramDic[k]))))
                else:
                    segFeatVecN4.append(float(0.0))

        perSegFeatVecProb.append(segFeatVecProb)
        perSegFeatVecN2.append(segFeatVecN2);
        perSegFeatVecN3.append(segFeatVecN3)
        perSegFeatVecN4.append(segFeatVecN4)


        segFeatVecProb = []
        segFeatVecN2 = []
        segFeatVecN3 = []
        segFeatVecN4 = []


    print ("train-set created")

    return globalFeatVecProb,globalFeatVecN2,globalFeatVecN3,globalFeatVecN4,perSegFeatVecProb,perSegFeatVecN2,perSegFeatVecN3,perSegFeatVecN4

def createTestSetForUser(i=0,ngramStop=2):
    globalTrainCommandPosDic, globalTrain2NGramDic, globalTrain3NGramDic, globalTrain4NGramDic, globalTrainProbCommandDic,perSegTrainCommandPosDic, perSegTrain2NGramDic, perSegTrain3NGramDic, perSegTrain4NGramDic, perSegTrainProbCommandDic,userSegmentsDicCommands, userSegmentsDicProbabilityCommands, userSegmentsDic2NGram, userSegmentsDic3NGram, userSegmentsDic4NGram=dicGen.getDics(i)

    perSegFeatVecProb = []
    segFeatVecProb = []

    perSegFeatVecN2 = []
    segFeatVecN2 = []

    perSegFeatVecN3 = []
    segFeatVecN3 = []

    perSegFeatVecN4 = []
    segFeatVecN4 = []

    perSegNotExistDicProb=[]
    segNotExistFeatVecProb={}
    perSegNotExistDicN2 = []
    segNotExistFeatVecN2 = {}
    perSegNotExistDicN3 = []
    segNotExistFeatVecN3 = {}
    perSegNotExistDicN4 = []
    segNotExistFeatVecN4 = {}

    for j in range(100):
        segNotExistFeatVec = {}
        """for k in globalTrainCommandPosDic:
            if userSegmentsDicCommands[j].__contains__(k):
                segFeatVec.append(float(float(userSegmentsDicCommands[j][k]) / (float(globalTrainCommandPosDic[k]))))
            else:
                segFeatVec.append(float(0.0))
                if segNotExistFeatVec.__contains__(k)==False:
                    segNotExistFeatVec[k]=1
                else:
                    segNotExistFeatVec[k]+=1"""
        if ngramStop>=1:
            for k in globalTrainProbCommandDic:
                if userSegmentsDicProbabilityCommands[j].__contains__(k):
                    segFeatVecProb.append(float(float(userSegmentsDicProbabilityCommands[j][k]) / (float(globalTrainProbCommandDic[k]))))
                else:
                    segFeatVecProb.append(float(0.0))
                    if segNotExistFeatVecProb.__contains__(k)==False:
                        segNotExistFeatVecProb[k]=1
                    else:
                        segNotExistFeatVecProb[k]+=1
        if ngramStop>=2:
            for k in globalTrain2NGramDic:
                if userSegmentsDic2NGram[j].__contains__(k):
                    segFeatVecN2.append(float(float(userSegmentsDic2NGram[j][k]) / (float(globalTrain2NGramDic[k]))))
                else:
                    segFeatVecN2.append(float(0.0))
                    if segNotExistFeatVecN2.__contains__(k)==False:
                        segNotExistFeatVecN2[k]=1
                    else:
                        segNotExistFeatVecN2[k]+=1

        if ngramStop>=3:
            for k in globalTrain3NGramDic:
                if userSegmentsDic3NGram[j].__contains__(k):
                    segFeatVecN3.append(float(float(userSegmentsDic3NGram[j][k]) / (float(globalTrain3NGramDic[k]))))
                else:
                    segFeatVecN3.append(float(0.0))
                    if segNotExistFeatVecN3.__contains__(k)==False:
                        segNotExistFeatVecN3[k]=1
                    else:
                        segNotExistFeatVecN3[k]+=1

        if ngramStop>=4:
            for k in globalTrain4NGramDic:
                if userSegmentsDic4NGram[j].__contains__(k):
                    segFeatVecN4.append(float(float(userSegmentsDic4NGram[j][k]) / (float(globalTrain4NGramDic[k]))))
                else:
                    segFeatVecN4.append(float(0.0))
                    if segNotExistFeatVecN4.__contains__(k)==False:
                        segNotExistFeatVecN4[k]=1
                    else:
                        segNotExistFeatVecN4[k]+=1
        perSegNotExistDicProb.append(segNotExistFeatVecProb)
        perSegNotExistDicN2.append(segNotExistFeatVecN2)
        perSegNotExistDicN3.append(segNotExistFeatVecN3)
        perSegNotExistDicN4.append(segNotExistFeatVecN4)

        perSegFeatVecProb.append(segFeatVecProb)
        perSegFeatVecN2.append(segFeatVecN2)
        perSegFeatVecN3.append(segFeatVecN3)
        perSegFeatVecN4.append(segFeatVecN4)

        segFeatVecProb = []
        segFeatVecN2=[]
        segFeatVecN3=[]
        segFeatVecN4=[]
    print ("test-set created")
    return  perSegFeatVecProb,perSegNotExistDicProb,perSegFeatVecN2,perSegNotExistDicN2,perSegFeatVecN3,perSegNotExistDicN3,perSegFeatVecN4,perSegNotExistDicN4

def createTestSetForUserKnownVsUnknown (i=0,ngramStop=2):

    globalTrainCommandPosDic, globalTrain2NGramDic, globalTrain3NGramDic, globalTrain4NGramDic, globalTrainProbCommandDic, perSegTrainCommandPosDic, perSegTrain2NGramDic, perSegTrain3NGramDic, perSegTrain4NGramDic, perSegTrainProbCommandDic, userSegmentsDicCommands, userSegmentsDicProbabilityCommands, userSegmentsDic2NGram, userSegmentsDic3NGram, userSegmentsDic4NGram = dicGen.getDics(i)



    perSegExistDist={}
    perSegNoExistDist={}
    for j in range(100):
        perSegExistDist[j]=[]
        perSegNoExistDist[j]=[]

        existProbDistDic = {}
        noExistProbDistDic = {}

        exist2NGramDistDic = {}
        noExist2NGramDistDic = {}

        exist3NGramDistDic = {}
        noExist3NGramDistDic = {}

        exist4NGramDistDic = {}
        noExist4NGramDistDic = {}


        """for k in globalTrainCommandPosDic:
            if userSegmentsDicCommands[j].__contains__(k):
                segFeatVec.append(float(float(userSegmentsDicCommands[j][k]) / (float(globalTrainCommandPosDic[k]))))
            else:
                segFeatVec.append(float(0.0))
                if segNotExistFeatVec.__contains__(k)==False:
                    segNotExistFeatVec[k]=1
                else:
                    segNotExistFeatVec[k]+=1"""
        if ngramStop>=1:
            for k in userSegmentsDicProbabilityCommands[j]:
                if globalTrainProbCommandDic.__contains__(k)==True:
                   if existProbDistDic.__contains__(k)==False:
                       existProbDistDic[k]=1
                   else:
                       existProbDistDic[k]+=1
                else:
                    if noExistProbDistDic.__contains__(k) == False:
                        noExistProbDistDic[k] = 1
                    else:
                        noExistProbDistDic[k] += 1
            for k in existProbDistDic:
                existProbDistDic[k]/=100.0
            for k in noExistProbDistDic:
                noExistProbDistDic[k]/=100.0
            perSegExistDist[j].append(existProbDistDic)
            perSegNoExistDist[j].append(noExistProbDistDic)

        if ngramStop>=2:
            for k in userSegmentsDic2NGram[j]:
                if globalTrain2NGramDic.__contains__(k):
                    if exist2NGramDistDic.__contains__(k) == False:
                        exist2NGramDistDic[k] = 1
                    else:
                        exist2NGramDistDic[k] += 1
                else:
                    if noExist2NGramDistDic.__contains__(k) == False:
                        noExist2NGramDistDic[k] = 1
                    else:
                        noExist2NGramDistDic[k] += 1
            for k in exist2NGramDistDic:
                exist2NGramDistDic[k]/=99.0
            for k in noExist2NGramDistDic:
                noExist2NGramDistDic[k]/=99.0
            perSegExistDist[j].append(exist2NGramDistDic)
            perSegNoExistDist[j].append(noExist2NGramDistDic)


        if ngramStop>=3:
            for k in userSegmentsDic3NGram[j]:
                if globalTrain3NGramDic.__contains__(k):
                    if exist3NGramDistDic.__contains__(k) == False:
                        exist3NGramDistDic[k] = 1
                    else:
                        exist3NGramDistDic[k] += 1
                else:
                    if noExist3NGramDistDic.__contains__(k) == False:
                        noExist3NGramDistDic[k] = 1
                    else:
                        noExist3NGramDistDic[k] += 1

            for k in exist3NGramDistDic:
                exist3NGramDistDic[k]/=98.0
            for k in noExist3NGramDistDic:
                noExist3NGramDistDic[k]/=98.0
            perSegExistDist[j].append(exist3NGramDistDic)
            perSegNoExistDist[j].append(noExist3NGramDistDic)

        if ngramStop>=4:
            for k in userSegmentsDic4NGram[j]:
                if globalTrain4NGramDic.__contains__(k):
                    if exist4NGramDistDic.__contains__(k) == False:
                        exist4NGramDistDic[k] = 1
                    else:
                        exist4NGramDistDic[k] += 1
                else:
                    if noExist4NGramDistDic.__contains__(k) == False:
                        noExist4NGramDistDic[k] = 1
                    else:
                        noExist4NGramDistDic[k] += 1
            for k in exist4NGramDistDic:
                exist4NGramDistDic[k]/=97.0
            for k in noExist4NGramDistDic:
                noExist4NGramDistDic[k]/=97.0
            perSegExistDist[j].append(exist4NGramDistDic)
            perSegNoExistDist[j].append(noExist4NGramDistDic)

    print ("test-set exist/no-exist dist dics were created")
    return perSegExistDist,perSegNoExistDist

#globalFeatVec,perSegFeatVecTrain=createTrainSetForUser(0)
#perSegFeatVecTest,segNotExistFeatVec=createTestSetForUser(0)
print("finished dataset-gen")