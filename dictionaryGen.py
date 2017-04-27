import sys
import time
from itertools import repeat
import io
import csv
import numpy as np

def createCommandDic ():

    for i in range(1):
        with io.open('E:/challenge/training_data/User'+str(i), 'rt',encoding="utf8") as file:

            for j,row in enumerate(file.readlines()):
                if j>=5000:
                    continue
                if commandDic.__contains__(row[:-1]+"_"+str(j%100))==False:
                    commandDic[row[:-1]+"_"+str(j%100)]=1
                else:
                    commandDic[row[:-1]+"_"+str(j%100)]+=1


def createCommandProbabilityDic ():

    for i in range(1):
        with io.open('E:/challenge/training_data/User'+str(i), 'rt',encoding="utf8") as file:
            commandProbabilityDic={}
            commandPDic={}
            for j,row in enumerate(file.readlines()):
                if j>=5000:
                    continue
                if commandPDic.__contains__(row[:-1])==False:
                    commandPDic[row[:-1]]=1
                else:
                    commandPDic[row[:-1]]+=1
    commandProbabilityDic=commandPDic
    return commandProbabilityDic

def createNGram2Dic ():
    for i in range(1):
        with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

            lines=file.readlines()
            counter = 1

            for row in range(len(lines)):
                if counter==5001:
                    break
                if counter%100==0:
                    counter+=1
                    continue
                if ngram2Dic.__contains__(lines[row][:-1] + "_" + lines[row+1][:-1]) == False:
                    ngram2Dic[lines[row][:-1] + "_" + lines[row+1][:-1]] = 1
                else:
                    ngram2Dic[lines[row][:-1] + "_" + lines[row+1][:-1]] += 1
                counter += 1
def createNGram3Dic ():
    for i in range(1):
        with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

            lines=file.readlines()
            counter = 1

            for row in range(len(lines)):
                if counter==5001:
                    break
                if (counter+1)%100==0 or counter%100==0:
                    counter+=1
                    continue
                if ngram3Dic.__contains__(lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]) == False:
                    ngram3Dic[lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]] = 1
                else:
                    ngram3Dic[lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]] += 1
                counter += 1

def createNGram4Dic ():
    for i in range(1):
        with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

            lines=file.readlines()
            counter = 1

            for row in range(len(lines)):
                if counter==5001:
                    break
                if (counter+2)%100==0 or (counter+1)%100==0 or counter%100==0:
                    counter+=1
                    continue
                if ngram4Dic.__contains__(lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]+ "_" + lines[row+3][:-1]) == False:
                    ngram4Dic[lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]+ "_" + lines[row+3][:-1]] = 1
                else:
                    ngram4Dic[lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]+ "_" + lines[row+3][:-1]] += 1
                counter += 1

def testUser ():

    for i in range(1):
        with io.open('E:/challenge/training_data/User'+str(i), 'rt',encoding="utf8") as file:
            userSegmentsDicCommands={}
            commandDicTest={}

            for j,row in enumerate(file.readlines()):
                if j<5000:
                    continue
                if commandDicTest.__contains__(row[:-1]+"_"+str(j%100))==False:
                    commandDicTest[row[:-1]+"_"+str(j%100)]=1
                else:
                    commandDicTest[row[:-1]+"_"+str(j%100)]+=1
                if (j-99)%100==0:
                    userSegmentsDicCommands[int((j-5000+1)/100)]=commandDicTest
                    commandDicTest={}


    for i in range(1):
        with io.open('E:/challenge/training_data/User'+str(i), 'rt',encoding="utf8") as file:
            userSegmentsDicProbabilityCommands={}
            commandDicTest={}

            for j,row in enumerate(file.readlines()):
                if j<5000:
                    continue
                if commandDicTest.__contains__(row[:-1])==False:
                    commandDicTest[row[:-1]]=1
                else:
                    commandDicTest[row[:-1]]+=1
                if (j-99)%100==0:
                    userSegmentsDicProbabilityCommands[int((j-5000+1)/100)]=commandDicTest
                    commandDicTest={}

    for i in range(1):
        with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:


                userSegmentsDic2NGram = {}
                ngram2DicTest={}
                fileLines=file.readlines()
                for j, row in enumerate(fileLines):
                    if j < 5000:
                        continue
                    if  (j - 99) % 100 == 0:
                        userSegmentsDic2NGram[int((j - 5000+1) / 100)] = ngram2DicTest
                        ngram2DicTest={}
                        continue
                    if ngram2DicTest.__contains__(fileLines[j][:-1] + "_" + fileLines[j + 1][:-1]) == False:
                        ngram2DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1]] = 1
                    else:
                        ngram2DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1]] += 1





    for i in range(1):
        with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

            userSegmentsDic3NGram  = {}
            ngram3DicTest={}
            fileLines=file.readlines()
            for j, row in enumerate(fileLines):
                if j < 5000:
                    continue
                if  (j-98)%100==0 or (j-99)%100==0:
                    if (j-99)%100==0:
                        userSegmentsDic3NGram [int((j - 5000+1) / 100)] = ngram3DicTest
                        ngram3DicTest={}
                    continue
                if ngram3DicTest.__contains__(fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][
                                                                                                  :-1]) == False:
                    ngram3DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][:-1]] = 1
                else:
                    ngram3DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][:-1]] += 1



    for i in range(1):
        with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

            userSegmentsDic4NGram  = {}
            ngram4DicTest={}
            fileLines = file.readlines()
            for j, row in enumerate(fileLines):
                if j < 5000:
                    continue
                if  (j - 97) % 100 == 0 or (j - 98) % 100 == 0 or (j - 99) % 100 == 0:
                    if (j-99)%100==0:
                        userSegmentsDic4NGram[int((j - 5000+1) / 100)] = ngram4DicTest
                        ngram4DicTest={}
                    continue
                if ngram4DicTest.__contains__(fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][
                                                                                                  :-1] + "_" + fileLines[j + 3][:-1]) == False:
                    ngram4DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][:-1] + "_" + fileLines[j + 3][:-1]] = 1
                else:
                    ngram4DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][:-1] + "_" + fileLines[j + 3][:-1]] += 1

    return userSegmentsDicCommands,userSegmentsDicProbabilityCommands,userSegmentsDic2NGram,userSegmentsDic3NGram,userSegmentsDic4NGram






commandDic={}
commandProbabilityDic={}
ngram2Dic={}
ngram3Dic={}
ngram4Dic={}

commandDicTest={}
ngram2DicTest={}
ngram3DicTest={}
ngram4DicTest={}

userSegmentsDicCommands = {}
userSegmentsDicProbabilityCommands={}
userSegmentsDic2NGram = {}
userSegmentsDic3NGram = {}
userSegmentsDic4NGram = {}

createCommandDic()
createNGram2Dic()
createNGram3Dic()
createNGram4Dic()
commandProbabilityDic=createCommandProbabilityDic()
userSegmentsDicCommands,userSegmentsDicProbabilityCommands,userSegmentsDic2NGram,userSegmentsDic3NGram,userSegmentsDic4NGram=testUser()


def checkDics():
    sum=0
    for k in userSegmentsDicCommands:
        for kk in userSegmentsDicCommands[k]:
            sum+=userSegmentsDicCommands[k][kk]
    print(sum)

    sum=0
    for k in userSegmentsDicProbabilityCommands:
        for kk in userSegmentsDicProbabilityCommands[k]:
            sum+=userSegmentsDicProbabilityCommands[k][kk]
    print(sum)

    sum=0
    for k in userSegmentsDic2NGram:
        for kk in userSegmentsDic2NGram[k]:
            sum+=userSegmentsDic2NGram[k][kk]
    print(sum)

    sum=0
    for k in userSegmentsDic3NGram:
        for kk in userSegmentsDic3NGram[k]:
            sum+=userSegmentsDic3NGram[k][kk]
    print(sum)

    sum=0
    for k in userSegmentsDic4NGram:
        for kk in userSegmentsDic4NGram[k]:
            sum+=userSegmentsDic4NGram[k][kk]
    print(sum)


    sum=0
    for k in commandDic:
        sum+=commandDic[k]
    print(sum)

    sum=0
    for k in commandProbabilityDic:
        sum+=commandProbabilityDic[k]
    print(sum)

    sum=0
    for k in ngram2Dic:
        sum+=ngram2Dic[k]
    print(sum)

    sum=0
    for k in ngram3Dic:
        sum+=ngram3Dic[k]
    print(sum)

    sum=0
    for k in ngram4Dic:
        sum+=ngram4Dic[k]
    print(sum)






print("finished")