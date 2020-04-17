
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np #klestionable 

depth = 0
def LOG(message):
    print(" "*depth*2 + str(message))


def getNumericNormalized(NUMERIC_FEATURES,train_file_path, prefix = ""):
    global depth
    depth += 1
    LOG("Reading csv values...")
    desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES]
    LOG("Getting information about dataset...")
    desc = desc.describe()
    mean = np.array(desc.T['mean'])
    std = np.array(desc.T['std'])
    LOG("Preparing records...")
    data = []
    with open(train_file_path, 'rt') as file:
        header = file.readline().strip().split(',')
        selected_columns = []
        for f in NUMERIC_FEATURES:
            selected_columns.append(header.index(f))
        for line in file:
            line = line.strip().split(',')
            data.append([])
            i = 0
            while i < len(selected_columns):
                val = float(line[selected_columns[i]])
                val = (val - mean[i]) / (std[i] + 0.0000001)
                data[len(data)-1].append(val)
                i += 1
    LOG("Done")
    depth -= 1
    return np.array(data)

def getCategoric(NUMERIC_FEATURES, filepath, prefix = ""):
    global depth
    depth += 1
    data = []
    LOG("Preparing records...")
    with open(filepath, 'rt') as file:
        header = file.readline().strip().split(',')
        selected_columns = list(range(len(header)))
        for f in NUMERIC_FEATURES:
            selected_columns.remove(header.index(f))
        for line in file:
            line = line.strip().split(',')
            data.append([])
            for i in selected_columns:
                val = float(line[i])
                data[len(data)-1].append(val)
    LOG("Done")
    depth -= 1
    return np.array(data)

def getOneColumn(name, filepath, prefix = ""):
    global depth
    depth += 1
    data = []
    LOG("Preparing records...")
    with open(filepath, 'rt') as file:
        header = file.readline().strip().split(',')
        selected_column = header.index(name)
        
        for line in file:
            line = line.strip().split(',')
            val = int(line[selected_column])
            data.append(val)
    LOG("Done")
    depth -= 1
    return np.array(data)


def csvToData(NUMERIC_FEATURES = ['age', 'TSH', 'T3', 'TT4', 'FTI', 'T4U','referral_source'],y_column = 'class',  filepath = 'data.csv', prefix = ""):
    global depth
    depth += 1
    LOG("Getting normalized numeric data...")
    normData = getNumericNormalized(NUMERIC_FEATURES, filepath,prefix+"  ")
    LOG("Getting categorical data...")
    catData = getCategoric(NUMERIC_FEATURES+[y_column], filepath,prefix+"  ")
    LOG("merging values...")
    x_train = np.concatenate((normData,catData),axis=1)
    LOG("Getting y values...")
    y_train = getOneColumn(y_column,filepath,prefix+"  ")
    LOG("Done")
    depth -= 1
    return x_train, y_train #as numpy

np.set_printoptions(precision=3, suppress=True)
LOG("Getting data from csv...")
x,y = csvToData(prefix="  ")
LOG("Found " + str(len(x)) + " records")
LOG("Creating datasets...")
tensor_x = torch.Tensor(x)
tensor_y = torch.from_numpy(y)

dataset = data.TensorDataset(tensor_x,tensor_y)
trainlength = int(len(dataset)*0.9)
train, test  = data.random_split(dataset,[trainlength, len(dataset)-trainlength])

trainset = data.DataLoader(train,batch_size=4,shuffle=True)
testset = data.DataLoader(test,batch_size=4,shuffle=True)
LOG("Done")
for data in trainset:
    break
x,y = data[0][0], data[1][0]
LOG("Example data tensor:")
LOG(x)
LOG("Example outcome:")
LOG(y)
LOG(data[0][0].shape) ##???????
LOG(data[1][0].shape)
total = 0
counter_dict = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0}

LOG("Balance:")
for data in trainset:
    _,ys = data
    for y in ys:
        counter_dict[int(y)] +=1
LOG(counter_dict)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(22,32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,32)
        self.fc4 = nn.Linear(32,14)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x,dim=1)
LOG("Creating network...")
net = Net()
Epoch = 10

optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
LOG("LEARNING")
depth += 1
for e in range(Epoch):
    LOG("Epoch: " + str(e))
    for data in trainset:
        X,y = data
        net.zero_grad()
        output = net(X)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    LOG(loss)
depth -= 1

correct = 0
total = 0
with torch.no_grad():
    for data in testset:
        X,y = data
        output = net(X)
        for idx,i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct +=1
            total += 1

print ("correct: " + str(correct) + ", out of: " + str(total))
