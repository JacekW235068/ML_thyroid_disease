import csv
import re


def getClassNames(iv_path):
    toRet = []
    with open(iv_path, 'r') as file:
        whole = file.read()
        lines = whole.split('\n')
        dash_filter = re.compile(r"\|.*")
        space_filter = re.compile(r" +")
        for curr_line in lines:
            curr_line = curr_line.rstrip()
            if len(curr_line) != 0:
                curr_line = dash_filter.sub('', curr_line)
                curr_line = space_filter.sub('_', curr_line)
                toRet.append(curr_line)

    if len(toRet) == 0:
        print('Failed to read data')
    else:
        return toRet


def getAppendList(listClasses: list):
    length = len(listClasses)
    toRet = []
    for i in range(length):
        toRet.append(0)
    return toRet


def getClassIndex(listHeader: list):
    i = 0
    for word in listHeader:
        word = word.rstrip()
        if word == "class":
            return i
        i = i + 1
    return -1


def genOutputList(listLine: list, listClassVal: list, iClassIndex: int):
    classes = listLine[iClassIndex].split(' ')
    for index in classes:
        listClassVal[int(index)] = 1

    retList = listLine + listClassVal
    return retList


def listToString(inputList: list):
    toRet = ''
    for element in inputList[:-1]:
        toRet += str(element) + ','
    toRet += str(inputList[-1])
    return toRet


classNames = getClassNames('class.meta')

with open('data.dat', 'r') as src:
    header = src.readline().split(',')
    classIndex = getClassIndex(header)
    if classIndex == -1:
        print("Could not find proper header. First line in file: \n")
        print(header)
        exit(-1)
    header = header + classNames
    del header[classIndex]
    with open('tonetwork.csv', 'w') as output:
        output.write(listToString(header) + '\n')
        for line in src:
            line = line.rstrip().split(',')
            appendList = getAppendList(classNames)
            toWrite = genOutputList(line, appendList, classIndex)
            del toWrite[classIndex]
            output.write(listToString(toWrite) + '\n')
