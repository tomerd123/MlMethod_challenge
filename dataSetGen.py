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
    """for k in globalTrainCommandPosDic:
        globalFeatVec.append(float(float(globalTrainCommandPosDic[k])/50.0))"""
    for k in globalTrainProbCommandDic:
        globalFeatVec.append(float(float(globalTrainProbCommandDic[k])/5000.0))
    for k in globalTrain2NGramDic:
        globalFeatVec.append(float(float(globalTrain2NGramDic[k])/(99.0*50.0)))
    """for k in globalTrain3NGramDic:
        globalFeatVec.append(float(float(globalTrain3NGramDic[k]) / (98.0 * 50.0)))
    for k in globalTrain4NGramDic:
        globalFeatVec.append(float(float(globalTrain4NGramDic[k]) / (97.0 * 50.0)))"""

    perSegFeatVec=[]
    segFeatVec=[]

    for j in range(50):
        """for k in globalTrainCommandPosDic:
            if perSegTrainCommandPosDic[j].__contains__(k):
                segFeatVec.append(float(float(perSegTrainCommandPosDic[j][k]) / (float(globalTrainCommandPosDic[k]))))
            else:
                segFeatVec.append(float(0.0))"""

        for k in globalTrainProbCommandDic:
            if perSegTrainProbCommandDic[j].__contains__(k):
                segFeatVec.append(
                    float(float(perSegTrainProbCommandDic[j][k]) / ( float(globalTrainProbCommandDic[k]))))
            else:
                segFeatVec.append(float(0.0))

        for k in globalTrain2NGramDic:
            if perSegTrain2NGramDic[j].__contains__(k):
                segFeatVec.append(
                    float(float(perSegTrain2NGramDic[j][k]) / (float( globalTrain2NGramDic[k]))))
            else:
                segFeatVec.append(float(0.0))

        """for k in globalTrain3NGramDic:
            if perSegTrain3NGramDic[j].__contains__(k):
                segFeatVec.append(
                    float(float(perSegTrain3NGramDic[j][k]) / (float(globalTrain3NGramDic[k]))))
            else:
                segFeatVec.append(float(0.0))

        for k in globalTrain4NGramDic:
            if perSegTrain4NGramDic[j].__contains__(k):
                segFeatVec.append(
                    float(float(perSegTrain4NGramDic[j][k]) / (float(globalTrain4NGramDic[k]))))
            else:
                segFeatVec.append(float(0.0))"""

        perSegFeatVec.append(segFeatVec)
        segFeatVec = []


    print ("train-set created")
    return globalFeatVec,perSegFeatVec

def createTestSetForUser(i=0):
    globalTrainCommandPosDic, globalTrain2NGramDic, globalTrain3NGramDic, globalTrain4NGramDic, globalTrainProbCommandDic,perSegTrainCommandPosDic, perSegTrain2NGramDic, perSegTrain3NGramDic, perSegTrain4NGramDic, perSegTrainProbCommandDic,userSegmentsDicCommands, userSegmentsDicProbabilityCommands, userSegmentsDic2NGram, userSegmentsDic3NGram, userSegmentsDic4NGram=dicGen.getDics(i)

    perSegFeatVec = []
    segFeatVec = []

    perSegNotExistDic=[]
    segNotExistFeatVec={}

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

        for k in globalTrainProbCommandDic:
            if userSegmentsDicProbabilityCommands[j].__contains__(k):
                segFeatVec.append(float(float(userSegmentsDicProbabilityCommands[j][k]) / (float(globalTrainProbCommandDic[k]))))
            else:
                segFeatVec.append(float(0.0))
                if segNotExistFeatVec.__contains__(k)==False:
                    segNotExistFeatVec[k]=1
                else:
                    segNotExistFeatVec[k]+=1

        for k in globalTrain2NGramDic:
            if userSegmentsDic2NGram[j].__contains__(k):
                segFeatVec.append(float(float(userSegmentsDic2NGram[j][k]) / (float(globalTrain2NGramDic[k]))))
            else:
                segFeatVec.append(float(0.0))
                if segNotExistFeatVec.__contains__(k)==False:
                    segNotExistFeatVec[k]=1
                else:
                    segNotExistFeatVec[k]+=1


        """for k in globalTrain3NGramDic:
            if userSegmentsDic3NGram[j].__contains__(k):
                segFeatVec.append(float(float(userSegmentsDic3NGram[j][k]) / (float(globalTrain3NGramDic[k]))))
            else:
                segFeatVec.append(float(0.0))
                if segNotExistFeatVec.__contains__(k)==False:
                    segNotExistFeatVec[k]=1
                else:
                    segNotExistFeatVec[k]+=1


        for k in globalTrain4NGramDic:
            if userSegmentsDic4NGram[j].__contains__(k):
                segFeatVec.append(float(float(userSegmentsDic4NGram[j][k]) / (float(globalTrain4NGramDic[k]))))
            else:
                segFeatVec.append(float(0.0))
                if segNotExistFeatVec.__contains__(k)==False:
                    segNotExistFeatVec[k]=1
                else:
                    segNotExistFeatVec[k]+=1"""
        perSegNotExistDic.append(segNotExistFeatVec)
        perSegFeatVec.append(segFeatVec)
        segFeatVec = []
    print ("test-set created")
    return  perSegFeatVec,perSegNotExistDic

globalFeatVec,perSegFeatVecTrain=createTrainSetForUser(0)
perSegFeatVecTest,segNotExistFeatVec=createTestSetForUser(0)
print("finished dataset-gen")