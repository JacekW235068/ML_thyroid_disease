import ann.ann_network, torch, numpy
import os

def dataset_splitted_half(x_tensor, y_tensor, size):
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    half_mark = int(len(dataset) * 0.5) 
    checksum = len(dataset) % 2 
    half_mark += checksum 
    
    train, test = torch.utils.data.random_split(dataset, [half_mark, len(dataset) - half_mark ])
    trainset = torch.utils.data.DataLoader(train, batch_size=size, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=size, shuffle=True)
    return trainset, testset

def genConfusionMatrix(clLabels, momentumRate, trainSet, testSet, neuronCount, countFeatures):
    ann.ann_network.LOG(" --- NOWA SIEC --- ")
    confMatrix = []
    for elem in clLabels:
        confMatrix.append( [ 0 ] * len(clLabels)  )
    hit = 0
    total = 0
    curr_net = ann.ann_network.Net(countFeatures, neuronCount)
    sgd_optm = torch.optim.SGD( curr_net.parameters(), lr = 0.001, momentum = momentumRate)
    for epoch in range(10):
        ann.ann_network.LOG("--- Epoch " + str(epoch+1) + "/10 ---")
        loss = None
        for currData in trainSet:
            x, y = currData
            curr_net.zero_grad()
            result = curr_net(x)
            loss = torch.nn.functional.nll_loss( result, y )
            loss.backward()
            sgd_optm.step()
    ann.ann_network.LOG("--- LOSS: " + str(loss.item() ) + " ---")
    with torch.no_grad():
        for currTest in testSet:
            x, y = currTest
            output = curr_net(x)
            for y_index,i in enumerate(output):
                predicted = torch.argmax( i ) 
                real = y [ y_index ] 
                confMatrix[ predicted ][ real ] += 1
                if torch.argmax( i ) == y[y_index]:
                            hit += 1
                total += 1

    perc = (hit*100)/total
    ann.ann_network.LOG("--- HIT/TOTAL PERC: " + str(perc) + " ---" ) 
    return perc, confMatrix

def train_network(clList, trainSet, testSet, listNeurons, stepMomentum, epochCount, countFeatures):
    listResults = []
    Result = tuple() 
    ann.ann_network.depth += 1
    for currNeuron in listNeurons:
        currMomentum = 0
        while currMomentum <= 0.99:
            ann.ann_network.LOG("--- Tworzenie sieci z liczba neuronow: " + str(currNeuron) +  " ---")
            curr_net = ann.ann_network.Net(countFeatures,  currNeuron ) 
            ann.ann_network.depth += 1
            ann.ann_network.LOG("--- Momentum: " + str(currMomentum) + ", Liczba neuronów: "+ str(currNeuron) + 
                      ", Stopień uczenia: 0.001  ---")
            currOptimizer = torch.optim.SGD( curr_net.parameters(), lr = 0.001, momentum=currMomentum )
            ann.ann_network.depth += 1
            for currEpoch in range(epochCount):
                ann.ann_network.LOG("--- Epoch: " + str(currEpoch+1) + "/" + str(epochCount) + " ---")
                loss = None
                for currData in trainSet:
                    x, y = currData
                    curr_net.zero_grad()
                    result = curr_net(x)
                    loss = torch.nn.functional.nll_loss( result, y )
                    loss.backward()
                    currOptimizer.step()
            ann.ann_network.depth -= 1
            ann.ann_network.LOG("--- LOSS: " + str(loss.item()) + " ---")
            clearClassList( clList )
            hit = 0
            total = 0
            with torch.no_grad():
                for currData in testSet:
                    x, y = currData
                    output = curr_net( x ) 
                    for y_index,i in enumerate(output):
                        if torch.argmax( i ) == y[y_index]:
                            clList[ y[y_index] ][2] += 1 
                            hit += 1
                        total += 1
                        clList[ y[y_index] ][1] += 1

                perc = (hit*100)/total
                ann.ann_network.LOG("--- HIT/TOTAL PERC: " + str(perc) + " ---" ) 
                printClassInfo( clList ) 
                Result = [ loss.item(), 0.001, currMomentum, currNeuron, hit, total, epochCount, countFeatures ]
                for elem in clList:
                    Result = Result + [ elem[1], elem[2] ] 

                listResults.append( Result )
                currMomentum += stepMomentum
            ann.ann_network.depth -= 1
    ann.ann_network.depth -= 1
    return listResults 
#   loss, learningrate, momentum, neuron, hit, total, epoch 
def saveResults( szPath : str, lResults : list, classList):
    with open(szPath, "w") as f:
        f.write("LOSS,LEARNING_RATE,MOMENTUM,NEURON_COUNT,HIT,TOTAL,EPOCH,FEATURE_COUNT,")
        for elem in classList[:-1]:
            f.write(elem[0]+'_totals,')
            f.write(elem[0]+'_hits,')

        f.write(classList[-1][0]+'_totals,')
        f.write(classList[-1][0]+'_hit\n')

        for result in lResults:
            for element in result[:-1]:
                f.write( str(element)+",") 
            f.write( str(result[-1])+"\n" ) 
    

def  filterFeatures( listNeeded : list, x_data):
    allLabels = ann.ann_network.NUMERIC_FEATURES +  ann.ann_network.getCategoricLabels(filepath="data.csv")
    toRet = []
    for loop_idx in range( len(x_data )):
        toAdd = []
        for feature in listNeeded:
            idx = allLabels.index(feature)
            toAdd.append( x_data[loop_idx][idx] )
        toRet.append(toAdd)
    return numpy.array(toRet)

import re 
def clearClassList(classList):
    for elem in classList:
        elem[1] = 0
        elem[2] = 0
    return

def getClassList(szClassMeta):
    toRet = []
    with open(szClassMeta, "r") as f:
        while True:
            line = f.readline()
            if line == "":
                break

            clLine = re.split('\\|' , line)
            clLine[0] = clLine[0].replace(' ', '_')
            toRet.append( [clLine[0], 0, 0] )

    return toRet

def printConfusionMatrix(confMatrx):
    for line in confMatrx:
        oneline = ""
        for elem in line:
            oneline = oneline + '\t' + str(elem)
        ann.ann_network.LOG(oneline)

def printClassInfo(classList):
    for elem in classList:
        ann.ann_network.LOG(" =============================================================================")
        ann.ann_network.LOG("Klasa: " + elem[0] + "\t Total: " + str(elem[1]) + "\t Hits: " + str(elem[2]) )
        ann.ann_network.LOG("Perc: " + str( float(elem[2]) * 100 /float(elem[1])) )
    ann.ann_network.LOG(" =============================================================================")
    return
import sys 
def disable_output():
    devNull = open(os.devnull, "w")
    sys.stdout = devNull

def launchConfusion(szDestination, neuronCount, momentum, listFeatures):
    clLabels = getClassList('class.meta')
    ann.ann_network.LOG("--- Pobieranie danych ---")
    x_csv, y_csv = ann.ann_network.csvToData()
    results = []
    avg_rest = []
    avg_perc = []

    for i in range(5):
        ann.ann_network.LOG("--- Tworzenie neuronów we/wy ---")
        x_filtered = filterFeatures( listFeatures , x_csv)
        x_tensor = torch.Tensor(x_filtered)
        y_tensor = torch.from_numpy(y_csv)
        train, test = dataset_splitted_half(x_tensor, y_tensor, 4)
        perc, matrix = genConfusionMatrix( clLabels, momentum, train, test, neuronCount, len(x_filtered[0]) )
        results.append(matrix)  
        avg_perc.append(perc)


    for i in range( len(results[0]) ):
        avg_rest.append( [0] * len(results[0][0]) )

    for result in results:
        for row in range( len(result) ):
            for column in range( len(result[row]) ):
                avg_rest[row][column] += result[row][column]

    for row in range( len(avg_rest) ):
        for column in range( len(avg_rest[row]) ):
            avg_rest[row][column] = float(avg_rest[row][column])/5

    local = 0
    for perc in avg_perc:
        local += perc 
    avg_percent = local / 5

    return avg_percent, avg_rest 


def start( szDestination, listNeurons, listFeatures, stepMomentum ):
    ann.ann_network.LOG("--- Tworzenie folderów ---")

    if not os.path.isdir( szDestination ):
         os.makedirs( szDestination )
         os.makedirs( szDestination+"/test" )
         os.makedirs( szDestination+"/train" )

    ann.ann_network.LOG("--- Pobieranie danych ---")
    x_csv, y_csv = ann.ann_network.csvToData()
    clList = getClassList('class.meta')
    ann.ann_network.LOG("--- Filtruj cechy ---")
    featuresAction = [   
               'TSH', 'FTI', 'TT4', 'T3',
               'T4U', 'age', 'on_thyroxine'
               ,'referral_source','pregnant',
               'sex', 'tumor', 'query_hyperthyroid', 
               'query_hypothyroid', 'thyroid_surgery', 
               'psych', 'sick', 'query_on_thyroxine', 
               'on_antithyroid_medication']
    iterate = 18 
    for feature in listFeatures:
        featuresAction.append( feature )
        ann.ann_network.LOG("--- Filtrowane cechy " + str(featuresAction) + " ---")
        x_filtered = filterFeatures( featuresAction , x_csv)
        # ----------------------------------
        # Tworzenie neuronów
        ann.ann_network.LOG("--- Tworzenie neuronów we/wy ---")
        x_tensor = torch.Tensor(x_filtered)
        y_tensor = torch.from_numpy(y_csv)
        # ---------------------------------
        # Tworzenie datasetu podzielonego na pół
        ann.ann_network.LOG("--- Dzielenie danych na pół ---")
        train, test = dataset_splitted_half(x_tensor, y_tensor, 4)
        # ---------------------------------
        # Trenowanie sieci
        ann.ann_network.LOG("--- Trenowanie sieci na danych trenujących i testowanie na testujących ---")
        train_wtrain_res = train_network( clList, test, train, listNeurons, stepMomentum, 10, len(x_filtered[0]) ) 
        ann.ann_network.LOG("--- Trenowanie sieci na danych testujących i testowanie na trenujących ---")
        train_wtest_res = train_network( clList, train, test, listNeurons, stepMomentum, 10, len(x_filtered[0]) )
        ann.ann_network.LOG("--- Koniec trenowania, zapisywanie ---")
        saveResults(szDestination + "/train/output_test_" + str(iterate) + ".csv", train_wtrain_res, clList )
        saveResults(szDestination + "/test/output_train_" + str(iterate) + ".csv", train_wtest_res, clList )
        train_wtrain_res = None  # Hehe GCed
        train_wtest_res = None 
        iterate += 1
