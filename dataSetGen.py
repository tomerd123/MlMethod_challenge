import sys
import time
from itertools import repeat
import io
import csv
import numpy as np
import dictionaryGen as dicGen


def createTrainSetForUser(i=0):
    globalTrainCommandPosDic, globalTrain2NGramDic, globalTrain3NGramDic, globalTrain4NGramDic, globalTrainProbCommandDic,perSegTrainCommandPosDic, perSegTrain2NGramDic, perSegTrain3NGramDic, perSegTrain4NGramDic, perSegTrainProbCommandDic,userSegmentsDicCommands, userSegmentsDicProbabilityCommands, userSegmentsDic2NGram, userSegmentsDic3NGram, userSegmentsDic4NGram=dicGen.getDics(i)

    globalFeatVec=[]
    for k in globalTrainCommandPosDic:
        globalFeatVec.append(float(float(globalTrainCommandPosDic[k])/50.0))
    for k in globalTrainProbCommandDic:
        globalFeatVec.append(float(float(globalTrainProbCommandDic[k])/5000.0))
    for k in globalTrain2NGramDic:
        globalFeatVec.append(float(float(globalTrain2NGramDic[k])/(99.0*50.0)))
    for k in globalTrain3NGramDic:
        globalFeatVec.append(float(float(globalTrain3NGramDic[k]) / (98.0 * 50.0)))
    for k in globalTrain4NGramDic:
        globalFeatVec.append(float(float(globalTrain4NGramDic[k]) / (97.0 * 50.0)))

    perSegFeatVec=[]
    segFeatVec=[]

    for j in range(50):
        for k in perSegTrainCommandPosDic[j]:
            segFeatVec.append(float(float(perSegTrainCommandPosDic[j][k])/100.0))

        for k in perSegTrainProbCommandDic[j]:
            segFeatVec.append(float(float(perSegTrainProbCommandDic[j][k])/100.0))

        for k in perSegTrain2NGramDic[j]:
            segFeatVec.append(float(float(perSegTrain2NGramDic[j][k]) / 99.0))

        for k in perSegTrain3NGramDic[j]:
            segFeatVec.append(float(float(perSegTrain3NGramDic[j][k]) / 98.0))

        for k in perSegTrain4NGramDic[j]:
            segFeatVec.append(float(float(perSegTrain4NGramDic[j][k]) / 97.0))

        perSegFeatVec.append(segFeatVec)
        segFeatVec=[]
    print ("train-set created")
    return globalFeatVec,perSegFeatVec

def createTestSetForUser(i=0):
    globalTrainCommandPosDic, globalTrain2NGramDic, globalTrain3NGramDic, globalTrain4NGramDic, globalTrainProbCommandDic,perSegTrainCommandPosDic, perSegTrain2NGramDic, perSegTrain3NGramDic, perSegTrain4NGramDic, perSegTrainProbCommandDic,userSegmentsDicCommands, userSegmentsDicProbabilityCommands, userSegmentsDic2NGram, userSegmentsDic3NGram, userSegmentsDic4NGram=dicGen.getDics(i)

    perSegFeatVec = []
    segFeatVec = []

    for j in range(100):
        for k in userSegmentsDicCommands[j]:
            segFeatVec.append(float(float(userSegmentsDicCommands[j][k]) / 100.0))

        for k in userSegmentsDicProbabilityCommands[j]:
            segFeatVec.append(float(float(userSegmentsDicProbabilityCommands[j][k]) / 100.0))

        for k in userSegmentsDic2NGram[j]:
            segFeatVec.append(float(float(userSegmentsDic2NGram[j][k]) / 99.0))

        for k in userSegmentsDic3NGram[j]:
            segFeatVec.append(float(float(userSegmentsDic3NGram[j][k]) / 98.0))

        for k in userSegmentsDic4NGram[j]:
            segFeatVec.append(float(float(userSegmentsDic4NGram[j][k]) / 97.0))

        perSegFeatVec.append(segFeatVec)
        segFeatVec = []
    print ("test-set created")
    return  perSegFeatVec

globalFeatVec,perSegFeatVec=createTrainSetForUser()
perSegFeatVecTest=createTestSetForUser()
print("finished dataset-gen")