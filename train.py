import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision.models as models
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import neptune.new as neptune

device = "mps"
# comment the above line and uncomment the line below if using CUDA enabled GPU
# device = "cuda"
# uncomment the line below if using cpu
# device = "cpu"

randomState = 420
learningRate = 0.0001
trainingEpochs = 15
batchSizeTrain = 8
batchSizeTest = 4

APIKEY = "YOUR API KEY"

run = neptune.init_run(
    project="r3yna/dogsvscats",
    api_token=APIKEY,
)

pathToDataset = './data/'
imagePath = []
labels = []
for folder in os.listdir(pathToDataset):
    for images in os.listdir(os.path.join(pathToDataset,folder)):
        image = os.path.join(pathToDataset,folder,images)
        imagePath.append(image)
        labels.append(folder)
data = {'Images':imagePath, 'Labels':labels}
data = pd.DataFrame(data)
#encoding text values as number 
labelEncoder = LabelEncoder()
data['EncodedLabels'] = labelEncoder.fit_transform(data['Labels'])
print("---------Encoded Label Ouputs---------")
print("Encoder label: 0\t Real label:",labelEncoder.inverse_transform([0])[0])
print("Encoder label: 1\t Real label:",labelEncoder.inverse_transform([1])[0])

print("---------Train Test Split---------")
data = data.sample(frac=1, random_state=randomState).reset_index()
dataTrain = data[0:int(0.9*len(data))]
dataTrain = dataTrain.sample(frac=1).reset_index(drop=True)
dataTest = data[int(0.9*len(data)):]
dataTest = dataTest.sample(frac=1).reset_index(drop=True)
print("Size of training set:",len(dataTrain))
print("Size of testing set:",len(dataTest))

class CustomDataset(Dataset):
    def __init__(self, pathToImages, labels, transform=None):
        self.pathToImages = pathToImages
        self.transform = transform
        self.labels = labels
        
    def __len__(self):
        return len(self.pathToImages)
    
    def __getitem__(self, index):
        image = Image.open(self.pathToImages[index])
        # converts images to rgb
        image = image.convert('RGB')
        # resizing image to 224x224
        image = image.resize((224,224))
        # convert the numerical labels to tensor
        label = torch.tensor(self.labels[index])
        # apply augmentations and transformation on the image
        if self.transform is not None:
            image = self.transform(image)
        return image, label

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

datasetTrain = CustomDataset(dataTrain['Images'], dataTrain['EncodedLabels'], transform)
datasetTest = CustomDataset(dataTest['Images'], dataTest['EncodedLabels'], transform)

trainLoader = torch.utils.data.DataLoader(datasetTrain, batch_size=batchSizeTrain)
testLoader = torch.utils.data.DataLoader(datasetTest, batch_size=batchSizeTest)

### since we only have two output classes, we change the output fc of the original model to give only two outputs
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.fc = torch.nn.Linear(512,2)
model.to(device)
print(model)

params = {
    "datasetRandomState": randomState, 
    "lr": learningRate,
    "trainingEpochs": trainingEpochs,
    "trainingBatchSize": 128,
    "inputSize": 224 * 224 * 3,
    "numberOfClasses": 2,
}
run["parameters"] = params

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

def evaluateModel(model):
    totalT=0
    correctT=0
    model.eval()
    with torch.no_grad():
        for dataT, targetT in (testLoader):
            dataT, targetT = dataT.to(device), targetT.to(device)
            outputT = model(dataT)
            _, predictionT = torch.max(outputT, dim=1)
            correctT += torch.sum(predictionT==targetT).item()
            totalT += targetT.size(0)
        valiationAccuracy = 100 * (correctT / totalT)
    return valiationAccuracy

summaryInterval = 20
validationAccuracyMax = 0.0
totalSteps = len(trainLoader)
for epoch in range(1, trainingEpochs+1):
    print(f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        #logs training loss at every step
        run["train/loss"].append(loss)
        optimizer.step()
        if(batch_idx)%summaryInterval==0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch, trainingEpochs, batch_idx, totalSteps, loss.item()))
    validationAccuracy = evaluateModel(model)
    print("Completed training for epoch")
    print("Accuracy on validation set:", validationAccuracy)
    #logs validation accuracy
    run["validation/accuracy"].append(validationAccuracy)
    if validationAccuracyMax<=validationAccuracy:
        validationAccuracyMax = validationAccuracy
        torch.save(model.state_dict(), 'classificationModel.pt')
        print('Detected network improvement, saving current model')
    model.train()

run.stop()