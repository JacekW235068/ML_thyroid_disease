from os import listdir
from os.path import isfile, join
import re
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
                        expline = expline.string[0:expline.string.find(":")]
                        expline = expline.replace(' ','_')
                        if expline not in valuetypeslist:
                            valuetypeslist.append(expline)
                            countAdded += 1
            print("File: \033[91m" + fileName + "\033[39m contains \033[92m" + str(count) + "\033[39m value types including \033[92m" + str(countAdded) + "\033[39m new.")
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
            print("File: \033[91m" + fileName + "\033[39m contains \033[92m" + str(count) + "\033[39m classes including \033[92m" + str(countAdded) + "\033[39m new.")
    return classlist


def countClasses(path, classlist, all = False):
    classcount = [0] * len(classlist) #apparently this is completely normal code in python
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for fileName in onlyfiles:
        if ".data" in fileName:
            with open(path + fileName, "r") as file:
                currentfilelasscount = [0] * len(classlist)
                for line in file:
                    for i in range(len(classlist)):
                        if classlist[i] in line:
                            classcount[i] += 1
                            currentfilelasscount[i] += 1
            print ("File \033[91m" + fileName + "\033[39m contains:")
            for i in range(len(classlist)):
                if currentfilelasscount[i] > 0 or all:
                    print ("\t" + classlist[i] + " - \033[92m" + str(currentfilelasscount[i]) + "\033[39m")
    return classcount

dir = "thyroid-disease/"
values = findValueTypes(dir)
print("-----------------------------------------------------------")
classes = findClasses(dir)
print("-----------------------------------------------------------")
classcount = countClasses(dir, classes)
