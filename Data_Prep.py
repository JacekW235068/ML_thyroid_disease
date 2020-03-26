from os import listdir
from os.path import isfile, join
import sys, os
import re
import random
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
        fout.write(value.split('|')[0])
        fout.write(',')
    fout.write('class')
    fout.write('\n')

class_in_data_regex = re.compile(r".*,(.*)\.|\d+")
def find_class_in_data_line(line, classes):
    class_search = class_in_data_regex.search(line)
    if class_search:
        return classes.index(class_search.group(1))

def createDataset(path, minimalCasesCount = 1, targetCasesCount = 500):
    classes = findClasses(path)
    classcount = countClasses(dir, classes)
    fout = open("data.dat", "wt")
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    lines_per_class = list(map(lambda _: [], classes))

    write_dataset_header(path, fout)

    for fileName in onlyfiles:
        if ".data" in fileName or ".test" in fileName:
            with open(path + fileName, "r") as file:
                for line in file:
                    class_index = find_class_in_data_line(line, classes)
                    lines_per_class[class_index].append(line.split('|')[0] + '\n') # cut the strange number at end of data line

    all_lines_of_dataset = []
    for class_index, lines_of_class in enumerate(lines_per_class):
        class_name = classes[class_index]
        if len(lines_of_class) < minimalCasesCount:
            print("Minimal cases count not reached for class {}".format(class_name))
            continue
        elif len(lines_of_class) >= targetCasesCount:
            print("Class {} extends the target, picking random {} cases.".format(class_name, targetCasesCount))
            all_lines_of_dataset.extend(random.sample(lines_of_class, targetCasesCount))
        else:
            print("{} cases of class {} will be randomly copied to fulfill {} target cases.".format(len(lines_of_class), class_name, targetCasesCount))
            for _ in range(targetCasesCount):
                all_lines_of_dataset.append(random.choice(lines_of_class));

    random.shuffle(all_lines_of_dataset)
    for line in all_lines_of_dataset:
        fout.write(line)

dir = "thyroid-disease/"
createMETAvalues(dir)
createMETAclasses(dir)
createDataset(dir)
# values = findValueTypes(dir)
# print("-----------------------------------------------------------")
# classes = findClasses(dir)
# print("-----------------------------------------------------------")
# classcount = countClasses(dir, classes)
# print("-----------------------------------------------------------")
# for i in range(len(classes)):
#     if classcount[i] > 0 or all:
#         print (classes[i] + " - \033[92m" + str(classcount[i]) + "\033[39m")