from deepAutoEncoderClass import DeepAutoencoder
import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import numpy as np

model = torch.load("savedModels/newModel.pt")
model.eval()



from datasetClass import myDataset
dataset = myDataset(csv_file = "nonSegmentedImageset.csv", root_dir='cityDataset', transform=transforms.ToTensor())

batch_size = 32;
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [800, 200])
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True);
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True);

encodedData = []
imageSize = 500
for idx in range(800):
    entry = train_dataset[idx][0][0].reshape(-1, imageSize * imageSize)
    encodedData.append(model.getEncoded(entry).detach().numpy())

temp = encodedData[0:]
x = [];
y = [];
z = [];
for entry in temp:
    x.append(entry[0][0])
    y.append(entry[0][1])
    z.append(entry[0][2])

plt.scatter(x, y, z)

encodedData = []
for idx in range(200):
    entry = test_dataset[idx][0][0].reshape(-1, imageSize * imageSize)
    encodedData.append(model.getEncoded(entry).detach().numpy())

temp = encodedData[0:]
x = [];
y = [];
z = [];
for entry in temp:
    x.append(entry[0][0])
    y.append(entry[0][1])
    z.append(entry[0][2])

plt.scatter(x, y, z)


plt.show()




