import ann.ann_menu


np.set_printoptions(precision=3, suppress=True)
LOG("Getting data from csv...")
x,y = csvToData(prefix="  ")

LOG("Feature importances: ")
for label, importance in sorted(zip(featureImportances(x, y), NUMERIC_FEATURES + getCategoricLabels("data.csv")), key=first):
    LOG("{}: {}".format(label, importance))

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

if __name__ == '__main__':
    ann.ann_menu.menu_open()
