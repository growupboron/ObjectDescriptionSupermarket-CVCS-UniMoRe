import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import Resize
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import SKUDatasetGPU, TEST_TRANSFORM, TRAIN_TRANSFORM
import argparse
import os
import json
import ast

# Custom transformation class for ToTensor
class ToTensorTransform(nn.Module):
    def __init__(self):
        super(ToTensorTransform, self).__init__()

    def forward(self, sample):
        image, target = sample['image'], sample['target']
        return {'image': torch.tensor(image, dtype=torch.float32), 'target': target}

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
args = parser.parse_args()

# Set the number of GPUs and the current GPU rank
num_gpus = args.num_gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rank = 0  # Default rank for single-GPU training

# Initialize the distributed backend (if using multi-GPU training)
if torch.cuda.is_available() and num_gpus > 1:
    os.environ['RANK'] = str(rank)  # Set the RANK environment variable
    torch.distributed.init_process_group(backend='nccl')
    rank = torch.distributed.get_rank()

# Create an instance of the SKU110K dataset
train_transform = nn.Sequential(Resize((256, 256)), ToTensorTransform())
val_transform = nn.Sequential(Resize((256, 256)), ToTensorTransform())
train_dataset = SKUDatasetGPU(split='train', transform=TRAIN_TRANSFORM)
val_dataset = SKUDatasetGPU(split='val', transform=TEST_TRANSFORM)

# Create data loaders for training and validation
train_sampler = None
if num_gpus > 1:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=(train_sampler is None), num_workers=4, sampler=train_sampler)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# Define the model architecture (SSD)
model = ssdlite320_mobilenet_v3_large(pretrained=False)

# Move the model to the device
model = model.to(device)

# Use DistributedDataParallel for multi-GPU training
if num_gpus > 1:
    model = DDP(model)

# Define the optimizer and learning rate scheduler
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Define the loss function for object detection
criterion = nn.SmoothL1Loss(reduction='sum')

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)  # Update sampler for distributed training

    # Print GPU training started message
    if rank == 0:
        print("GPU training started")

    for images, targets in train_dataloader:
        print(f"Batch size: {len(images)}")
        # Move images to the device
        images = [image.to(device) for image in images]

        # Convert targets to dictionaries and move them to the device
        parsed_targets = []
        for t in targets:
            if isinstance(t, str):
                try:
                    parsed_targets.append(json.loads(t))
                except json.JSONDecodeError:
                    parsed_targets.append({})
            else:
                parsed_targets.append(t)

        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in parsed_targets
        ]

        # Ensure the length of targets is the same as images
        if len(images) != len(targets):
            continue

        # Forward pass
        print("Forward pass started")
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        print("Forward pass completed")

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for images, targets in val_dataloader:
            # Move images to the device
            images = [image.to(device) for image in images]

            # Convert targets to dictionaries and move them to the device
            parsed_targets = []
            for t in targets:
                if isinstance(t, str):
                    try:
                        parsed_targets.append(json.loads(t))
                    except json.JSONDecodeError:
                        parsed_targets.append({})
                else:
                    parsed_targets.append(t)

            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in parsed_targets
            ]

            # Ensure the length of targets is the same as images
            if len(images) != len(targets):
                continue

            # Forward pass
            outputs = model(images)

            # Compute validation loss or other metrics

    # Adjust learning rate
    lr_scheduler.step()

    # Print epoch statistics
    if rank == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
if rank == 0:
    torch.save(model.state_dict(), 'ssd_OD.pth')