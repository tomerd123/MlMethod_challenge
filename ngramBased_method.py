import sys
import time
from itertools import repeat
import io
import csv
import numpy as np
import dataSetGen as dsGen
import dA
import math
def trainAndExecute (i=0):
        scores=[]
        labels=[]

        for j2 in range(40):
            with io.open('E:/challenge/training_data/user2SegLabels.csv', 'rt', encoding="utf8") as file:
                file.readline()
                labList = []
                for j in range(150):

                    labList.append(file.readline().split(',')[j2])
                labels.append(labList)



        globalFeatVec, perSegFeatVecTrain = dsGen.createTrainSetForUser(0)
        perSegFeatVecTest, segNotExistFeatVec = dsGen.createTestSetForUser(0)


        da=dA.dA(n_visible=len(globalFeatVec),n_hidden=len(globalFeatVec)/3)

        print ("starting train\n")
        for iter in range(1):
            counter=0

            for k in range(len(perSegFeatVecTrain)):
                print(str(da.train(input=np.array(perSegFeatVecTrain[k])))+","+str(labels[i+1][counter]))
                counter+=1
            avgScores=0
            counter=0
            for k in range(len(perSegFeatVecTrain)):
                avgScores+=da.feedForward(input=np.array(perSegFeatVecTrain[k]))
                nnOutput=da.feedForward(input=np.array(perSegFeatVecTrain[k]))
                print(str(nnOutput)+","+str(labels[i+1][counter]))
                counter+=1
            avgScores/=len(perSegFeatVecTrain)

        print ("finished train, starts test\n")
        for k in range(len(perSegFeatVecTest)):
            notExistSum=0
            for x in segNotExistFeatVec[k]:
                notExistSum+=segNotExistFeatVec[k][x]
            notExistRatio=float(float(notExistSum)/float(len(globalFeatVec)))
            nnOutputDif=math.fabs(da.feedForward(input=np.array(perSegFeatVecTest[k]))-avgScores)
            print(str(nnOutputDif)+","+str(labels[i+1][counter]))
            counter+=1
            scores.append((nnOutputDif,notExistRatio))
        return scores
        print ("finished test\n")

trainAndExecute(0)