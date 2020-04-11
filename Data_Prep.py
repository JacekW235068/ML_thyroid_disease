from os import listdir
from os.path import isfile, join
import sys, os
import re
import random
import io
import statistics

def findValueTypes(path):
    valuetypeslist = []
    regexp1 = re.compile(r"[A-Z,a-z,',',\s,0-9]*\:\s*[A-Z,a-z,',',\s]*\.")
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for fileName in onlyfiles:
        if ".names" in fileName:
            with open(path + fileName, "r") as file:
                count = 0
                countAdded = 0
                for line in file:
                    expline = regexp1.search(line)
                    if expline:
                        count +=1
                        valtype = expline.string[expline.string.find(':')+1:expline.string.find('.')]
                        valtype = valtype.strip()
                        expline = expline.string[0:expline.string.find(":")]
                        expline = expline.replace(' ','_')
                        expline = expline + "|" + valtype
                        if expline not in valuetypeslist:
                            valuetypeslist.append(expline)
                            countAdded += 1
            print("File: \033[94m" + fileName + "\033[39m contains \033[92m" + str(count) + "\033[39m value types including \033[92m" + str(countAdded) + "\033[39m new.")
    return valuetypeslist

def findClasses(path):
    classlist = []
    regexp1 = re.compile(r"[A-Z,a-z,',',\s,0-9]*\:\s*[A-Z,a-z,',',\s]*\.")
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for fileName in onlyfiles:
        if ".names" in fileName:
            with open(path + fileName, "r") as file:
                count = 0
                countAdded = 0
                for line in file:
                    test = regexp1.search(line)
                    if not test and ';' not in line:
                        line = line.strip()
                        if '.' in line:
                            line = line[:line.index('.')]
                        for Class in line.split(','):
                            Class = Class.strip()
                            if  len(Class) != 0:
                                count += 1
                                if Class not in classlist:
                                    countAdded += 1
                                    classlist.append(Class)
            print("File: \033[94m" + fileName + "\033[39m contains \033[92m" + str(count) + "\033[39m classes including \033[92m" + str(countAdded) + "\033[39m new.")
    return classlist


def countClasses(path, classlist, all = False):
    classcount = [0] * len(classlist) #apparently this is completely normal code in python
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for fileName in onlyfiles:
        if ".data" in fileName or ".test" in fileName:
            with open(path + fileName, "r") as file:
                currentfilelasscount = [0] * len(classlist)
                for line in file:
                    for i in range(len(classlist)):
                        if classlist[i] in line:
                            classcount[i] += 1
                            currentfilelasscount[i] += 1
            print ("File \033[94m" + fileName + "\033[39m contains:")
            for i in range(len(classlist)):
                if currentfilelasscount[i] > 0 or all:
                    print ("\t" + classlist[i] + " - \033[92m" + str(currentfilelasscount[i]) + "\033[39m")
    return classcount

def createMETAvalues(path, silent = False):
    if silent:
        sys.stdout = open(os.devnull, 'w')
    print("Finding value types...")
    values = findValueTypes(path)
    print("Done.")
    print("Writing to \033[94mvlaues.meta\033[39m ...")
    fout = open("values.meta", "wt")
    for line in values:
        fout.write(line + "\n")
    print("Done.")
    if silent:
        sys.stdout = sys.__stdout__

def createMETAclasses(path, minimalCasesCount = -1, silent = False):
    if silent:
        sys.stdout = open(os.devnull, 'w')
    print("Finding classes...")
    classes = findClasses(path)
    print("Done.")
    print("Finding cases...")
    classcount = countClasses(dir, classes)
    print("Done.")
    print("Writing to \033[94mclass.meta\033[39m ...")
    fout = open("class.meta", "wt")
    for i in range(len(classes)):
        if classcount[i] > minimalCasesCount:
            fout.write(classes[i] + "|" + str(classcount[i]) + "\n")
    print("Done.")
    if silent:
        sys.stdout = sys.__stdout__

def write_dataset_header(path, fout):
    values = findValueTypes(path)
    for value in values:
        fout.write(value.split('|')[0].replace(" ", "_"))
        fout.write(',')
    fout.write('class')
    fout.write('\n')

class_in_data_regex = re.compile(r".*,(.*)\.|\d+")
def find_class_in_data_line(line, classes):
    class_search = class_in_data_regex.search(line)
    if class_search:
        return classes.index(class_search.group(1))

data_without_class_regex = re.compile(r"(.*),.*\.|\d+")
def get_data_without_class(line):
    search = data_without_class_regex.search(line)
    if search:
        return search.group(1)

def merge_data_from_all_files(path):
    classNames = findClasses(path)
    datafiles = [f for f in listdir(path) if isfile(join(path, f)) and ".data" in f]
    testfiles = [f for f in listdir(path) if isfile(join(path, f)) and ".test" in f]

    data_data = []
    data_classes = []
    test_data = []
    test_classes = []
    with open(path + datafiles[0], "r") as file:
        for line in file:
            data_data.append(get_data_without_class(line))
            data_classes.append(set())
    with open(path + testfiles[0], "r") as file:
        for line in file:
            test_data.append(get_data_without_class(line))
            test_classes.append(set())
    for fileName in datafiles:
        with open(path + fileName, "r") as file:
            for i, line in enumerate(file):
                data_classes[i].add(str(find_class_in_data_line(line, classNames)))
    for fileName in testfiles:
        with open(path + fileName, "r") as file:
            for i, line in enumerate(file):
                test_classes[i].add(str(find_class_in_data_line(line, classNames)))

    data = data_data + test_data
    classes = data_classes + test_classes
    negative_index = str(classNames.index("negative"))
    result = []
    for i, line in enumerate(data):
        if len(classes[i]) > 1 and negative_index in classes[i]:
            classes[i].remove(negative_index)
        result.append(line + "," + " ".join(classes[i]) + "\n")
    return result

def createDataset(path, minimalCasesCount = 1, targetCasesCount = 500):
    classes = findClasses(path)
    fout = open("data.dat", "wt")

    write_dataset_header(path, fout)
    merged_data = merge_data_from_all_files(path)

    for line in merged_data:
        fout.write(line)

def removeColumn(filepath, columnName):
    file = open(filepath, 'rt')
    if os.path.exists("temp.dat"):
        os.remove("temp.dat")
    outputFile = open ("temp.dat", 'wt')
    header = file.readline().strip().split(',')
    if columnName not in header:
        return
    columnIndex = header.index(columnName)
    del header[columnIndex]
    header = ','.join(header)
    outputFile.write(header + "\n")
    for line in file:
        line = line.split(',')
        del line[columnIndex]
        line = ','.join(line)
        outputFile.write(line)
    file.close()
    outputFile.close()
    # os.remove(filepath)
    # os.rename(r'temp.dat',filepath)

def findEmptyColumns(filepath,minimumPercentage = 1, emptychar = '?'):
    minimumPercentage = minimumPercentage/100.0
    linescount = 0
    file = open(filepath, 'rt')
    header = file.readline().strip().split(',')
    valuesCount = [0] * len(header)
    for line in file:
        linescount += 1
        line = line.strip().split(',')
        i = 0
        while i < len(line):
            if line[i] != emptychar:
                valuesCount[i] += 1
            i += 1
    file.close()
    i = 0
    empty = []
    while i < len(header):
        if valuesCount[i]/linescount <= minimumPercentage:
            empty.append(header[i])
        i += 1
    return empty

def findkeywordColumns(filepath, keyword):
    file = open(filepath, 'rt')
    header = file.readline().strip().split(',')
    file.close()
    i = 0
    columns = []
    while i < len(header):
        if keyword in header[i]:
            columns.append(header[i])
        i += 1
    return columns

def removeRangeColumns(filepath, columnNames):
    file = open(filepath, 'rt')
    if os.path.exists("temp.dat"):
        os.remove("temp.dat")
    outputFile = open ("temp.dat", 'wt')
    header = file.readline().strip().split(',')
    columnIndexes = []
    for columnName in columnNames:
        if columnName in header:
            columnIndexes.append(header.index(columnName))
    columnIndexes.sort()
    i = 0
    #this is readable btw
    while i < len(columnIndexes):
        columnIndexes[i] -= i
        i += 1
    for columnIndex in columnIndexes:
        del header[columnIndex]
    header = ','.join(header)
    outputFile.write(header + "\n")
    for line in file:
        line = line.split(',')
        for columnIndex in columnIndexes:
            del line[columnIndex]
        line = ','.join(line)
        outputFile.write(line)
    file.close()
    outputFile.close()
    # os.remove(filepath)
    # os.rename(r'temp.dat',filepath)

def getDataStatsForMocking(dataSet):
    columnIsNumeric = []
    categoricalData = []
    categoricalDataCount = []
    numericData = []
    with open("values.meta", "r") as file:
        for line in file:
            line = line[line.index('|')+1:]
            if "continuous" in line:
                columnIsNumeric.append(True)
            else:
                columnIsNumeric.append(False)
            numericData.append([])
            categoricalData.append([])
            categoricalDataCount.append([])
    with io.StringIO(dataSet) as file:
        for line in file:
            line = line.strip().split(',')
            i = 0
            while i < len(columnIsNumeric):
                if '?' in line[i]:
                    pass
                elif columnIsNumeric[i]:
                    value = float(line[i])
                    numericData[i].append(value)
                else:
                    if line[i] not in categoricalData[i]:
                         categoricalData[i].append(line[i])
                         categoricalDataCount[i].append(0)
                    categoricalDataCount[i][categoricalData[i].index(line[i])] += 1 #what the fuck is this
                i += 1
    numeric_variance = []
    numeric_avg = []
    for data in numericData:
        if len(data) > 2:
            numeric_variance.append(statistics.stdev(data))
            numeric_avg.append(sum(data)/len(data))
        else:
            numeric_variance.append(0)
            numeric_avg.append(0)
    return columnIsNumeric, categoricalData, categoricalDataCount, numeric_variance, numeric_avg

def sum(array):
    sum = 0
    for num in array:
        sum += num
    return sum

def mockValues(filepath, output):
    classes = []
    classNames = []
    classCases = []
    classStats = []
    with open("class.meta", "r") as file:
        i = 0
        for line in file:
            classes.append(str(i))
            classCases.append("")
            classStats.append([])
            classNames.append(line[:line.index('|')])
            i+=1
    with open(filepath, "r") as file:
        file.readline()
        for line in file:
            i = 0
            while i < len(classes):
                if classes[i] in line:
                    classCases[i] += line
                i+=1
    i = 0
    while i < len(classes):
        classStats[i] = getDataStatsForMocking(classCases[i])
        print("finished class stats for " + classNames[i])
        i += 1
    print("writing output file")
    if os.path.exists(output):
        os.remove(output)
    with open(filepath, "r") as file:
        with open(output, "wt") as outputFile:
            for line in file:
                i = 0
                while i < len(classes):
                    if classes[i] in line:
                        break
                    i+=1
                line = line.strip().split(',')
                j = 0
                while j < len(line)-1:
                    if '?' in line[j]:
                        if classStats[i][0][j]:
                            start = classStats[i][4][j] - classStats[i][3][j]
                            stop = classStats[i][4][j] + classStats[i][3][j]
                            line[j] = str(round(random.uniform(start, stop),3))
                        else:
                            sumOfCases = sum(classStats[i][2][j])
                            rand = random.random()*sumOfCases
                            k = 0
                            while rand > 0:
                                rand -= classStats[i][2][j][k]
                                k += 1
                            line[j] = classStats[i][1][j][k-1]
                    j +=1
                outputFile.write(','.join(line) + '\n')
    print("Done")

dir = "thyroid-disease/"
# removeRangeColumns('data.dat',['age','sex'])
# convertTextLabelsToNumbers('data.dat')
createMETAclasses(dir)
createMETAvalues(dir)
createDataset(dir)
mockValues('data.dat', 'output.dat')
# values = findValueTypes(dir)
# print("-----------------------------------------------------------")
# classes = findClasses(dir)
# print("-----------------------------------------------------------")
# classcount = countClasses(dir, classes)
# print("-----------------------------------------------------------")
# for i in range(len(classes)):
#     if classcount[i] > 0 or all:
#         print (classes[i] + " - \033[92m" + str(classcount[i]) + "\033[39m")
