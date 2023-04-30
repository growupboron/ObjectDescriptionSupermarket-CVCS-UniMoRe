import os.path

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from datasets import GroceryStoreDataset01, collate_fn, transform
from tqdm import tqdm

trainset = GroceryStoreDataset01(split='train', transform=transform)
testset = GroceryStoreDataset01(split='test', transform=transform)
valset = GroceryStoreDataset01(split='val', transform=transform)

# create a dataloader
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)
valloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# instantiate the model--> resnet18
model = torchvision.models.resnet18(pretrained=True).to(device)

# train and test the model

# define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# define the loss function
criterion = nn.CrossEntropyLoss()

# train the model
epochs = 25
losses = []
accuracies = []
if not os.path.exists("model.pth"):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')

        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # validate the model
            acc = 0
            if i % 10 == 0:
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
                    accuracies.append(acc)
                model.train()

            pbar.set_postfix({'loss': running_loss / (i + 1), 'accuracy': acc})

        losses.append(running_loss)

    print(''''
##################################################################################
#                                                                                #
#                       Training Completed                                       #
#                                                                                #
##################################################################################
''')
else:
    model.load_state_dict(torch.load("model.pth"))

# test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy}%')

print(''''
##################################################################################
#                                                                                #
#                       Testing Completed                                        #
#                                                                                #
##################################################################################
''')
# save the model
torch.save(model.state_dict(), "model.pth")


print(''''
##################################################################################
#                                                                                #
#                       Model Saved                                              #
#                                                                                #
##################################################################################
''')
'''plt.plot(losses)
plt.plot(accuracies)
plt.show()'''

