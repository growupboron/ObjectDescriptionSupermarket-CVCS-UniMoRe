import math
import os.path

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import GoogLeNet_Weights

from datasets import GroceryStoreDataset01, collate_fn, TRANSFORM
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

trainset = GroceryStoreDataset01(split='train', transform=TRANSFORM)
valset = GroceryStoreDataset01(split='val', transform=TRANSFORM)
testset = GroceryStoreDataset01(split='test', transform=TRANSFORM)
num_classes = len(trainset.classes)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=6)
valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=6)
testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=6)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}...')
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(device)
model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

if os.path.exists('classifier.pth'):
    model.load_state_dict(torch.load('classifier.pth'))
epochs = 20

print('''
#################################################################
#                                                               #
#                       Training Started                        #
#                                                               #                 
#################################################################
''')
# training loop
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')

    for i, data in enumerate(pbar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        pbar.set_postfix({'loss': running_loss / (i + 1)})
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        pbar.set_postfix({'val_acc': round(acc, 1)})
    model.train()
    pbar.set_postfix({'loss': running_loss, 'val_acc': round(acc, 1)})

print(''''
#################################################################
#                                                               #
#                       Training Completed                      #
#                                                               #                 
#################################################################
''')

torch.save(model.state_dict(), 'classifier.pth')

print('''
#################################################################
#                                                               #
#                       Model Saved                             #
#                                                               #                 
#################################################################
      ''')

# final test

print('Testing Started...')

correct = 0
test_pbar = tqdm(testloader, desc=f'Testing', unit='batch')

for test_data in test_pbar:
    test_inputs, test_targets = test_data[0].to(device), test_data[1].to(device)
    test_prediction = model(test_inputs)
    _, output = torch.max(test_prediction, 1)
    correct += output.eq(test_targets).sum().item()

accuracy = correct / len(testset)
print(f'Final Accuracy: {accuracy}')

