import sys
import time
from itertools import repeat
import io
import csv
import numpy as np

def createCommandDic ():

    for i in range(10):
        with io.open('E:/challenge/training_data/User'+str(i), 'rt',encoding="utf8") as file:

            counter=1
            for row in file:
                if commandDic.__contains__(row[:-1]+"_"+str(counter))==False:
                    commandDic[row[:-1]+"_"+str(counter)]=1
                else:
                    commandDic[row[:-1]+"_"+str(counter)]+=1
                counter+=1
                if counter==101:
                    counter=1

def createNGram2Dic ():
    for i in range(10):
        with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

            lines=file.readlines()
            counter = 1

            for row in range(len(lines)):
                if counter%100==0:
                    counter+=1
                    continue
                if ngram2Dic.__contains__(lines[row][:-1] + "_" + lines[row+1][:-1]) == False:
                    ngram2Dic[lines[row][:-1] + "_" + lines[row+1][:-1]] = 1
                else:
                    ngram2Dic[lines[row][:-1] + "_" + lines[row+1][:-1]] += 1
                counter += 1
def createNGram3Dic ():
    for i in range(10):
        with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

            lines=file.readlines()
            counter = 1

            for row in range(len(lines)):
                if (counter+1)%100==0 or counter%100==0:
                    counter+=1
                    continue
                if ngram3Dic.__contains__(lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]) == False:
                    ngram3Dic[lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]] = 1
                else:
                    ngram3Dic[lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]] += 1
                counter += 1

def createNGram4Dic ():
    for i in range(10):
        with io.open('E:/challenge/training_data/User' + str(i), 'rt', encoding="utf8") as file:

            lines=file.readlines()
            counter = 1

            for row in range(len(lines)):
                if (counter+2)%100==0 or (counter+1)%100==0 or counter%100==0:
                    counter+=1
                    continue
                if ngram4Dic.__contains__(lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]+ "_" + lines[row+3][:-1]) == False:
                    ngram4Dic[lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]+ "_" + lines[row+3][:-1]] = 1
                else:
                    ngram4Dic[lines[row][:-1] + "_" + lines[row+1][:-1]+ "_" + lines[row+2][:-1]+ "_" + lines[row+3][:-1]] += 1
                counter += 1


commandDic={}
ngram2Dic={}
ngram3Dic={}
ngram4Dic={}

createCommandDic()
createNGram2Dic()
createNGram3Dic()
createNGram4Dic()

for x in ngram4Dic:
    print(str(x)+"_"+str(ngram4Dic[x]))


print("finished")