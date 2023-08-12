import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from PIL import Image
from datasets import SKUDataset, TEST_TRANSFORM, TRAIN_TRANSFORM

# Define the transformations to apply to the dataset images
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any other necessary transformations such as normalization, resizing, etc.
])

# Create an instance of the SKU110K dataset
train_dataset = SKUDataset(split='train', transform=TRAIN_TRANSFORM)
val_dataset = SKUDataset(split='val', transform=TEST_TRANSFORM)

# Create data loaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Define the device to use (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architecture (SSD)
model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')


# Move the model to the device
model.to(device)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Define the loss function
criterion = utils.evaluate

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_dataloader:
        # Move images and targets to the device
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for images, targets in val_dataloader:
            # Move images and targets to the device
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(images)
            # Compute validation loss or other metrics

    # Adjust learning rate
    lr_scheduler.step()

    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'ssd_OD.pth')

# yolo version ---> same problem..
'''from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from PIL import Image
from datasets import SKUDataset, TEST_TRANSFORM, TRAIN_TRANSFORM
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
import torch
from tqdm import tqdm
from torchvision.models.detection import YOLOv3
from torchvision.models.detection.yolo import yolo_loss

# Create an instance of the SKU110K dataset
train_dataset = SKUDataset(split='train', transform=TRAIN_TRANSFORM)
val_dataset = SKUDataset(split='val', transform=TEST_TRANSFORM)

# Create data loaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Define the device to use (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architecture (YOLOv3)
model = YOLOv3(pretrained=False)

# Move the model to the device
model.to(device)

# Define the optimizer and learning rate scheduler
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        images, targets = batch

        # Move images and targets to the device
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        outputs = model(images)

        # Compute loss using the YOLO loss function
        loss = yolo_loss(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            images, targets = batch

            # Move images and targets to the device
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(images)

            # Compute validation loss
            val_loss += yolo_loss(outputs, targets).item()

    val_loss /= len(val_dataloader)

    # Adjust learning rate
    lr_scheduler.step()

    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'yolo_OD.pth')'''
