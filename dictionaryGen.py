import sys
import time
from itertools import repeat
import io
import csv
import numpy as np


#train by global trainset
def createCommandDic (i=0):


    with io.open('E:/challenge/training_data/User'+str(i), 'rt',encoding="utf8") as file:
        commandDic={}
        for j,row in enumerate(file.readlines()):
            if j>=5000:
                continue
            if commandDic.__contains__(row[:-1]+"_"+str(j%100))==False:
                commandDic[row[:-1]+"_"+str(j%100)]=1
            else:
                commandDic[row[:-1]+"_"+str(j%100)]+=1

        return commandDic
def createCommandProbabilityDic (i=0):


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

def createNGram2Dic (i=0):

    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:
        ngram2Dic={}
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
        return ngram2Dic
def createNGram3Dic (i=0):
    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

        lines=file.readlines()
        counter = 1
        ngram3Dic={}
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
        return ngram3Dic

def createNGram4Dic (i=0):

    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

        ngram4Dic={}
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
        return ngram4Dic

#train per segment
def createPerSegmentCommandDic (i=0):


    with io.open('E:/challenge/training_data/User'+str(i), 'rt',encoding="utf8") as file:
        userSegmentsDicCommands = {}
        perSegCommandDic={}
        for j,row in enumerate(file.readlines()):
            if j>=5000:
                continue
            if perSegCommandDic.__contains__(row[:-1]+"_"+str(j%100))==False:
                perSegCommandDic[row[:-1]+"_"+str(j%100)]=1
            else:
                perSegCommandDic[row[:-1]+"_"+str(j%100)]+=1

            if (j-99)%100==0:
                userSegmentsDicCommands[int((j)/100)]=perSegCommandDic
                perSegCommandDic={}
        return userSegmentsDicCommands

def createPerSementCommandProbabilityDic (i=0):

    with io.open('E:/challenge/training_data/User'+str(i), 'rt',encoding="utf8") as file:
        userSegmentCommandProbabilityDic={}
        commandProbPerSegmentDic={}
        for j,row in enumerate(file.readlines()):
            if j>=5000:
                continue
            if commandProbPerSegmentDic.__contains__(row[:-1])==False:
                commandProbPerSegmentDic[row[:-1]]=1
            else:
                commandProbPerSegmentDic[row[:-1]]+=1
            if (j - 99) % 100 == 0:
                userSegmentCommandProbabilityDic[int((j  ) / 100)] = commandProbPerSegmentDic
                commandProbPerSegmentDic = {}
    return userSegmentCommandProbabilityDic


def createPerSegmentNGram2Dic (i=0):
    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

        lines=file.readlines()
        perSeg2NGramDic={}
        userSegments2NGramDic={}

        for j,row in enumerate(lines):
            if j>=5000:
                continue
            if (j - 99) % 100 == 0:
                userSegments2NGramDic[int((j ) / 100)] = perSeg2NGramDic
                perSeg2NGramDic = {}
                continue

            if perSeg2NGramDic.__contains__(row[:-1] + "_" + lines[j+1][:-1]) == False:
                perSeg2NGramDic[row[:-1] + "_" + lines[j+1][:-1]] = 1
            else:
                perSeg2NGramDic[row[:-1] + "_" + lines[j+1][:-1]] += 1


        return userSegments2NGramDic

def createPerSegmentNGram3Dic (i=0):

    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

        lines=file.readlines()
        ngram3PerSegDic={}
        userSegments3NGramDic={}

        for j,row in enumerate(lines):
            if j>=5000:
                continue
            if (j - 98) % 100 == 0 or (j - 99) % 100 == 0:
                if (j - 99) % 100 == 0:
                    userSegments3NGramDic[int((j  ) / 100)] = ngram3PerSegDic
                    ngram3PerSegDic = {}
                continue
            if ngram3PerSegDic.__contains__(row[:-1] + "_" + lines[j+1][:-1]+ "_" + lines[j+2][:-1]) == False:
                ngram3PerSegDic[row[:-1] + "_" + lines[j+1][:-1]+ "_" + lines[j+2][:-1]] = 1
            else:
                ngram3PerSegDic[row[:-1] + "_" + lines[j+1][:-1]+ "_" + lines[j+2][:-1]] += 1

        return userSegments3NGramDic


def createPerSegmentNGram4Dic (i=0):
    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

        lines=file.readlines()
        userSegments4NGramDic={}
        ngram4PerSegDic={}

        for j,row in enumerate(lines):
            if j>=5000:
                continue
            if (j - 97) % 100 == 0 or (j - 98) % 100 == 0 or (j - 99) % 100 == 0:
                if (j - 99) % 100 == 0:
                    userSegments4NGramDic[int((j ) / 100)] = ngram4PerSegDic
                    ngram4PerSegDic = {}
                continue
            if ngram4PerSegDic.__contains__(row[:-1] + "_" + lines[j+1][:-1]+ "_" + lines[j+2][:-1]+ "_" + lines[j+3][:-1]) == False:
                ngram4PerSegDic[row[:-1] + "_" + lines[j+1][:-1]+ "_" + lines[j+2][:-1]+ "_" + lines[j+3][:-1]] = 1
            else:
                ngram4PerSegDic[row[:-1] + "_" + lines[j+1][:-1]+ "_" + lines[j+2][:-1]+ "_" + lines[j+3][:-1]] += 1


    return userSegments4NGramDic


#test per segment

def testUser (i=0):

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
                userSegmentsDicCommands[int((j-5000)/100)]=commandDicTest
                commandDicTest={}


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
                userSegmentsDicProbabilityCommands[int((j-5000)/100)]=commandDicTest
                commandDicTest={}

    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:


            userSegmentsDic2NGram = {}
            ngram2DicTest={}
            fileLines=file.readlines()
            for j, row in enumerate(fileLines):
                if j < 5000:
                    continue
                if  (j - 99) % 100 == 0:
                    userSegmentsDic2NGram[int((j - 5000) / 100)] = ngram2DicTest
                    ngram2DicTest={}
                    continue
                if ngram2DicTest.__contains__(fileLines[j][:-1] + "_" + fileLines[j + 1][:-1]) == False:
                    ngram2DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1]] = 1
                else:
                    ngram2DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1]] += 1





    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

        userSegmentsDic3NGram  = {}
        ngram3DicTest={}
        fileLines=file.readlines()
        for j, row in enumerate(fileLines):
            if j < 5000:
                continue
            if  (j-98)%100==0 or (j-99)%100==0:
                if (j-99)%100==0:
                    userSegmentsDic3NGram [int((j - 5000) / 100)] = ngram3DicTest
                    ngram3DicTest={}
                continue
            if ngram3DicTest.__contains__(fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][
                                                                                              :-1]) == False:
                ngram3DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][:-1]] = 1
            else:
                ngram3DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][:-1]] += 1



    with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

        userSegmentsDic4NGram  = {}
        ngram4DicTest={}
        fileLines = file.readlines()
        for j, row in enumerate(fileLines):
            if j < 5000:
                continue
            if  (j - 97) % 100 == 0 or (j - 98) % 100 == 0 or (j - 99) % 100 == 0:
                if (j-99)%100==0:
                    userSegmentsDic4NGram[int((j - 5000) / 100)] = ngram4DicTest
                    ngram4DicTest={}
                continue
            if ngram4DicTest.__contains__(fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][
                                                                                              :-1] + "_" + fileLines[j + 3][:-1]) == False:
                ngram4DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][:-1] + "_" + fileLines[j + 3][:-1]] = 1
            else:
                ngram4DicTest[fileLines[j][:-1] + "_" + fileLines[j + 1][:-1] + "_" + fileLines[j + 2][:-1] + "_" + fileLines[j + 3][:-1]] += 1

    return userSegmentsDicCommands,userSegmentsDicProbabilityCommands,userSegmentsDic2NGram,userSegmentsDic3NGram,userSegmentsDic4NGram



def getDics(i=0):

    #global train-set
    globalTrainCommandPosDic=createCommandDic(i)
    globalTrain2NGramDic=createNGram2Dic(i)
    globalTrain3NGramDic=createNGram3Dic(i)
    globalTrain4NGramDic=createNGram4Dic(i)
    globalTrainProbCommandDic=createCommandProbabilityDic(i)

    #per segment train-set

    perSegTrainCommandPosDic=createPerSegmentCommandDic(i)
    perSegTrain2NGramDic=createPerSegmentNGram2Dic(i)
    perSegTrain3NGramDic=createPerSegmentNGram3Dic(i)
    perSegTrain4NGramDic=createPerSegmentNGram4Dic(i)
    perSegTrainProbCommandDic=createPerSementCommandProbabilityDic(i)

    #per segment test-set
    userSegmentsDicCommands,userSegmentsDicProbabilityCommands,userSegmentsDic2NGram,userSegmentsDic3NGram,userSegmentsDic4NGram=testUser(i)

    return globalTrainCommandPosDic,globalTrain2NGramDic,globalTrain3NGramDic,globalTrain4NGramDic,globalTrainProbCommandDic,perSegTrainCommandPosDic,perSegTrain2NGramDic,perSegTrain3NGramDic,perSegTrain4NGramDic,perSegTrainProbCommandDic,userSegmentsDicCommands,userSegmentsDicProbabilityCommands,userSegmentsDic2NGram,userSegmentsDic3NGram,userSegmentsDic4NGram


def checkDics(i=0):
    #createDictionaries

    # global train-set
    globalTrainCommandPosDic = createCommandDic(i)
    globalTrain2NGramDic = createNGram2Dic(i)
    globalTrain3NGramDic = createNGram3Dic(i)
    globalTrain4NGramDic = createNGram4Dic(i)
    globalTrainProbCommandDic = createCommandProbabilityDic(i)

    # per segment train-set

    perSegTrainCommandPosDic = createPerSegmentCommandDic(i)
    perSegTrain2NGramDic = createPerSegmentNGram2Dic(i)
    perSegTrain3NGramDic = createPerSegmentNGram3Dic(i)
    perSegTrain4NGramDic = createPerSegmentNGram4Dic(i)
    perSegTrainProbCommandDic = createPerSementCommandProbabilityDic(i)

    # per segment test-set
    userSegmentsDicCommands, userSegmentsDicProbabilityCommands, userSegmentsDic2NGram, userSegmentsDic3NGram, userSegmentsDic4NGram = testUser(i)

    #test per segment
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

    #train per segment

    sum = 0
    for k in perSegTrainCommandPosDic:
        for kk in perSegTrainCommandPosDic[k]:
            sum += perSegTrainCommandPosDic[k][kk]
    print(sum)

    sum = 0
    for k in perSegTrainProbCommandDic:
        for kk in perSegTrainProbCommandDic[k]:
            sum += perSegTrainProbCommandDic[k][kk]
    print(sum)

    sum = 0
    for k in perSegTrain2NGramDic:
        for kk in perSegTrain2NGramDic[k]:
            sum += perSegTrain2NGramDic[k][kk]
    print(sum)

    sum = 0
    for k in perSegTrain3NGramDic:
        for kk in perSegTrain3NGramDic[k]:
            sum += perSegTrain3NGramDic[k][kk]
    print(sum)

    sum = 0
    for k in perSegTrain4NGramDic:
        for kk in perSegTrain4NGramDic[k]:
            sum += perSegTrain4NGramDic[k][kk]
    print(sum)

    #train global
    sum=0
    for k in globalTrainCommandPosDic:
        sum+=globalTrainCommandPosDic[k]
    print(sum)

    sum=0
    for k in globalTrainProbCommandDic:
        sum+=globalTrainProbCommandDic[k]
    print(sum)

    sum=0
    for k in globalTrain2NGramDic:
        sum+=globalTrain2NGramDic[k]
    print(sum)

    sum=0
    for k in globalTrain3NGramDic:
        sum+=globalTrain3NGramDic[k]
    print(sum)

    sum=0
    for k in globalTrain4NGramDic:
        sum+=globalTrain4NGramDic[k]
    print(sum)




checkDics()


print("finished dic-gen")