from os import listdir
from os.path import isfile, join
import re

typeslist = []
classlist = []
classcount = []
regexp1 = re.compile(r"[A-Z,a-z,',',\s,0-9]*\:\s*[A-Z,a-z,',',\s]*\.")
regexp2 = re.compile(r"[A-Z,a-z,',',\s,0-9]*\:")
mypath = "thyroid-disease/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for fileName in onlyfiles:
    if ".names" in fileName:
        print (fileName)
        with open(mypath + fileName, "r") as file:
            count = 0
            for line in file:
                expline = regexp1.search(line)
                if expline: 
                    count +=1
                    expline = expline.string[0:expline.string.find(":")]
                    expline = expline.replace(' ','_')
                    if expline not in typeslist:
                        typeslist.append(expline)
                else:
                    if ';' not in line:
                        line = line.strip()
                        if '.' in line:
                            line = line[:line.index('.')]
                        for Class in line.split(','):
                            if Class not in classlist:
                                classlist.append(Class.strip())
                                classcount.append(0)
            print (count)
classlist.remove('')
for fileName in onlyfiles:
    if ".data" in fileName:
        print (fileName)
        with open(mypath + fileName, "r") as file:
            for line in file:
                for i in range(len(classlist)):
                    if classlist[i] in line:
                        classcount[i] += 1
print("-------------")
for item in typeslist:
    print(item)
print(len(typeslist))

print("-------------")   
for i in range(len(classlist)):
    print (classlist[i] + " - " + str(classcount[i]))