
import math
import os
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import GoogLeNet_Weights
import sys
from datasets import GroceryStoreDataset01, collate_fn, TEST_TRANSFORM
from tqdm import tqdm


testset = GroceryStoreDataset01(split='test', transform=TEST_TRANSFORM)
num_classes = 81

testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=6)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}...')
# model = torchvision.models.efficientnet_b0(pretrained=True).to(device)
model = torchvision.models.densenet121(pretrained=True).to(device)
# model = torchvision.models.resnet18().to(device)
model.classifier = nn.Linear(
    1024,  num_classes).to(device)
criterion = nn.CrossEntropyLoss()

# load the classifier's weigths
if os.path.exists('classifier.pth'):
    model.load_state_dict(torch.load('classifier.pth'))
else:
    # error handling
    print("Error: weigths file not found.\nPlease run the train.py script before and then retry!", file=sys.stderr)
    exit(1)
# final test

print('**********Testing Started**********')

correct = 0
test_pbar = tqdm(testloader, desc=f'Testing', unit='batch')

for test_data in test_pbar:
    test_inputs, test_targets = test_data[0].to(device), test_data[1].to(device)
    test_prediction = model(test_inputs)
    _, output = torch.max(test_prediction, 1)
    correct += output.eq(test_targets).sum().item()

accuracy = round(correct*100 / len(testset), 1)
print(f'Final Accuracy: {accuracy}%')

print('**********Program Ended**********')
