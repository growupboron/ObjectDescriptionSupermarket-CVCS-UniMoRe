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


trainset = GroceryStoreDataset01(split='train', transform=TRANSFORM)
valset = GroceryStoreDataset01(split='val', transform=TRANSFORM)
testset = GroceryStoreDataset01(split='test', transform=TRANSFORM)
num_classes = len(trainset.classes)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}...')
#model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).to(device)

model = torchvision.models.resnet18().to(device)
model.fc = nn.Sequential(nn.Linear(
    model.fc.in_features,  num_classes).to(device),
    torch.nn.ReLU()
    )
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

if os.path.exists('classifier.pth'):
    model.load_state_dict(torch.load('classifier.pth'))
epochs = 40

print('''
#################################################################
#                                                               #
#                       Training Started                        #
#                                                               #                 
#################################################################
''')
losses = []
val_losses = []
acc = 0

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

        pbar.set_postfix({'loss': running_loss / (i + 1), 'val_accuracy': acc})
    if i == len(trainloader) - 1:
        if losses:
            if losses[-1] < running_loss:
                print("Possible overfit...")
        losses.append(running_loss/(i+1))
        
    model.eval()
    correct = 0
    val_loss = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(valloader):
            images, val_labels = data
            images = images.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(images)
            _, predicted = torch.max(val_outputs.data, 1)
            val_loss += criterion(val_outputs, val_labels).item()

            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
        acc = 100 * correct / total
        val_loss /= idx+1
        val_losses.append(val_loss)
    
    model.train()
    

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
x_axis = np.linspace(1, epochs, epochs)
correct = 0
test_pbar = tqdm(testloader, desc=f'Testing', unit='batch')

for test_data in test_pbar:
    test_inputs, test_targets = test_data[0].to(device), test_data[1].to(device)
    test_prediction = model(test_inputs)
    _, output = torch.max(test_prediction, 1)
    correct += output.eq(test_targets).sum().item()

accuracy = round(correct*100 / len(testset), 1)
print(f'Final Accuracy: {accuracy}%')
fig, ax = plt.subplots()
ax.plot(x, losses, label="training loss")
ax.plot(x, val_losses, label="validation loss")
ax.legend()
plt.saveplot('train_results.png')
