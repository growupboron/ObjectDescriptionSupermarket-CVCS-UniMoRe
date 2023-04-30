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
epochs = 10
losses = []
accuracies = []
if not os.path.exists("model.pth"):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader)):
            inputs, labels = data
            # labels = labels.view(-1)
    
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
    
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
    
            if i % 2 == 0:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                losses.append(running_loss)
                running_loss = 0.0
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
    print('Accuracy of the network on the test images: %d {}' % accuracy)


# save the model
torch.save(model.state_dict(), "model.pth")

'''plt.plot(losses)
plt.plot(accuracies)
plt.show()'''

