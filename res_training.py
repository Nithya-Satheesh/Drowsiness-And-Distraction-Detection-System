
import torch,torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import cv2
import os

device='cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())

train_transform=transforms.Compose([
transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.RandomAffine(degrees=(25)),
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.Normalize((0.5,0.5,0.5),(1,1,1))
    ])
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize((0.5,0.5,0.5),(1,1,1))
])

train_data=torchvision.datasets.ImageFolder('dataset/dataset_/train',transform=train_transform)
test_data=torchvision.datasets.ImageFolder('dataset/dataset_/test',transform=transform)
val_data=torchvision.datasets.ImageFolder('dataset/dataset_/val',transform=transform)

classes=train_data.class_to_idx
print(classes)
print(len(classes))

batch_size=32
epochs=200
num_classes=4
load_model=True

train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=True)
val_loader=torch.utils.data.DataLoader(val_data,batch_size=batch_size,shuffle=True)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft = model_ft.to(device)
model_ft

loss_fn=nn.CrossEntropyLoss()
optimizer=optim.SGD(model_ft.parameters(),lr=0.0001) 

path="models"


for epoch in range(epochs):
    for i,(images,labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.to(device)
        outputs=model_ft(images)
        loss=loss_fn(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}],loss {}'.format(epoch+1,epochs,loss.item()))
    torch.save(model_ft.state_dict(),os.path.join(path,'epoch-{}.pt'.format(epoch)))



with torch.no_grad():   
    correct=0
    total=0
    for i,(images,labels) in enumerate(val_loader):
        images=images.to(device)
        labels=labels.to(device)
        outputs=model_ft(images)
        pred=torch.max(outputs,1)[1]
        total+=labels.size(0)
        correct += (pred == labels).sum().item()
        accu=correct/total
    print("Accuracy :%.2f"%accu)


