import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np #klestionable

from sklearn.ensemble import RandomForestClassifier

NUMERIC_FEATURES = ['age', 'TSH', 'T3', 'TT4', 'FTI', 'T4U','referral_source']
depth = 0

class Net(nn.Module):
    def __init__(self, start_count, count_neuron):
        super().__init__()
        self.fc1 = nn.Linear(start_count,count_neuron)
        self.fc2 = nn.Linear(count_neuron, count_neuron)
        self.fc3 = nn.Linear(count_neuron, 14)

    def forward(self,x):
        x = F.relu(self.fc1(x) )
        x = F.relu(self.fc2(x) )
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

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

def getCategoricLabels(filepath,NUMERIC_FEATURES = ['age', 'TSH', 'T3', 'TT4', 'FTI', 'T4U','referral_source']):
    with open(filepath, 'rt') as file:
        header = file.readline().strip().split(',')
        for f in NUMERIC_FEATURES:
            header.remove(f)
        return header

def featureImportances(x, y):
    clf = RandomForestClassifier()
    clf.fit(x, y)
    return clf.feature_importances_

def first(tup):
    return tup[0]




