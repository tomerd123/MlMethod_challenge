import sys
import time
from itertools import repeat
import io
import csv
import numpy as np
import dataSetGen as dsGen
import dA
import math
def trainAndExecute (i=0,ngramStop=2):
        scoresProb=[]
        scoresN2=[]
        scoresN3=[]
        scoresN4=[]
        labels=[]

        for j2 in range(40):
            with io.open('E:/challenge/training_data/user2SegLabels.csv', 'rt', encoding="utf8") as file:
                file.readline()
                labList = []
                for j in range(150):

                    labList.append(file.readline().split(',')[j2])
                labels.append(labList)

        globalFeatVecProb, globalFeatVecN2, globalFeatVecN3, globalFeatVecN4, perSegFeatVecProb,perSegFeatVecN2,perSegFeatVecN3,perSegFeatVecN4 = dsGen.createTrainSetForUser(i,ngramStop)
        userSegFeatVecProb, userSegNotExistDicProb, userSegFeatVecN2, userSegNotExistDicN2, userSegFeatVecN3, userSegNotExistDicN3, userSegFeatVecN4, userSegNotExistDicN4 = dsGen.createTestSetForUser(i,ngramStop)

        #train and test prob


        da=dA.dA(n_visible=len(perSegFeatVecProb[0]),n_hidden=len(perSegFeatVecProb[0])/3)

        print ("starting train\n")
        for iter in range(1):
            counter=0

            for k in range(len(perSegFeatVecProb)):
                da.train(input=np.array(perSegFeatVecProb[k]))
                counter+=1

        counter=0

        print ("finished train, starts test\n")


        for k in range(len(userSegFeatVecProb)):

            nnOutputProb=da.feedForward(input=np.array(userSegFeatVecProb[k]))
            scoresProb.append(nnOutputProb)
            counter+=1
        #train and test N2

        if len(perSegFeatVecN2[0])==0:
            print("dfdsfs")
        da = dA.dA(n_visible=len(perSegFeatVecN2[0]), n_hidden=len(perSegFeatVecN2[0]) / 3)

        print ("starting train\n")
        for iter in range(1):
            counter = 0

            for k in range(len(perSegFeatVecN2)):
                da.train(input=np.array(perSegFeatVecN2[k]))
                counter += 1

        counter = 0


        print ("finished train, starts test\n")

        for k in range(len(userSegFeatVecN2)):
            nnOutputN2 = da.feedForward(input=np.array(userSegFeatVecN2[k]))
            scoresN2.append(nnOutputN2)
            counter += 1
        #train and test N3
        if len(perSegFeatVecN3[0])==0:
            print("dfdsfs")
        da = dA.dA(n_visible=len(perSegFeatVecN3[0]), n_hidden=len(perSegFeatVecN3[0]) / 3)

        print ("starting train\n")
        for iter in range(1):
            counter = 0

            for k in range(len(perSegFeatVecN3)):
                da.train(input=np.array(perSegFeatVecN3[k]))
                counter += 1

        counter = 0


        print ("finished train, starts test\n")

        for k in range(len(userSegFeatVecN3)):
            nnOutputN3 = da.feedForward(input=np.array(userSegFeatVecN3[k]))
            scoresN3.append(nnOutputN3)
            counter += 1

        #train and test N4
        if len(perSegFeatVecN4[0])==0:
            print("dfdsfs")
        da = dA.dA(n_visible=len(perSegFeatVecN4[0]), n_hidden=len(perSegFeatVecN4[0]) / 3)

        print ("starting train\n")
        for iter in range(1):
            counter = 0

            for k in range(len(perSegFeatVecN4)):
                da.train(input=np.array(perSegFeatVecN4[k]))
                counter += 1

        counter = 0



        print ("finished train, starts test\n")

        for k in range(len(userSegFeatVecN4)):
            nnOutputN4 = da.feedForward(input=np.array(userSegFeatVecN4[k]))
            counter += 1
            scoresN4.append(nnOutputN4)
        print ("finished test\n")
        return scoresProb,scoresN2,scoresN3,scoresN4

#trainAndExecute(7,4)