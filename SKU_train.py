import torch
import time
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torch.nn import SmoothL1Loss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from datasets import SKUDataset, TEST_TRANSFORM, TRAIN_TRANSFORM

# Create an instance of the SKU110K dataset
train_dataset = SKUDataset(split='train', transform=TRAIN_TRANSFORM)
val_dataset = SKUDataset(split='val', transform=TEST_TRANSFORM)


# Create data loaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Define the model architecture (SSD)
model = ssdlite320_mobilenet_v3_large(pretrained=False)

# Move the model to the device
device = torch.device("cuda")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
model.to(device)

# Define the optimizer and learning rate scheduler
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Define the loss function
criterion = SmoothL1Loss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    start_time = time.time()  # Start time for epoch
    for images, x1, y1, x2, y2, class_id, image_width, image_height in train_dataloader:
        # Move images and targets to the device
        images = images.to(device)
        x1 = x1.to(device)
        y1 = y1.to(device)
        x2 = x2.to(device)
        y2 = y2.to(device)
        class_id = class_id.to(device)
        image_width = image_width.to(device)
        image_height = image_height.to(device)

        # Create a list of target dictionaries
        targets = []
        for i in range(len(images)):
            target = {}
            target['boxes'] = torch.tensor([[x1[i], y1[i], x2[i], y2[i]]], dtype=torch.float32).to(device)
            target['labels'] = torch.tensor([class_id[i]], dtype=torch.int64).to(device)
            target['image_width'] = image_width[i].to(device)
            target['image_height'] = image_height[i].to(device)
            targets.append(target)

        # Forward pass
        outputs = model(images, targets)

        # Print the outputs dictionary
        print(outputs)

        # Check if 'bbox_regression' and 'classification' keys exist in outputs dictionary
        if 'bbox_regression' in outputs and 'classification' in outputs:
            output_boxes = outputs['bbox_regression']
            output_labels = outputs['classification']
        else:
            raise KeyError("Keys 'bbox_regression' and 'classification' not found in outputs dictionary.")

        # Extract tensors from targets list
        target_boxes = torch.cat([target['boxes'] for target in targets])
        target_labels = torch.cat([target['labels'] for target in targets])

        loss = criterion(output_boxes, target_boxes) + criterion(output_labels, target_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    for images, x1, y1, x2, y2, class_id, image_width, image_height in val_dataloader:
        # Move images and targets to the device
        images = images.to(device)
        x1 = x1.to(device)
        y1 = y1.to(device)
        x2 = x2.to(device)
        y2 = y2.to(device)
        class_id = (class_id - class_id.min()).to(device)
        image_width = image_width.to(device)
        image_height = image_height.to(device)

        # Create a list of target dictionaries
        targets = []
        for i in range(len(images)):
            target = {}
            target['boxes'] = torch.tensor([[x1[i], y1[i], x2[i], y2[i]]], dtype=torch.float32).to(device)
            target['labels'] = torch.tensor([class_id[i]], dtype=torch.int64).to(device)
            target['image_width'] = image_width[i].to(device)
            target['image_height'] = image_height[i].to(device)
            targets.append(target)

        # Forward pass
        outputs = model(images, targets)

        # Check if 'bbox_regression' and 'classification' keys exist in outputs dictionary
        if 'bbox_regression' in outputs and 'classification' in outputs:
            output_boxes = outputs['bbox_regression']
            output_labels = outputs['classification']
        else:
            raise KeyError("Keys 'bbox_regression' and 'classification' not found in outputs dictionary.")

        # Extract tensors from targets list
        target_boxes = torch.cat([target['boxes'] for target in targets])
        target_labels = torch.cat([target['labels'] for target in targets])

        loss = criterion(output_boxes, target_boxes) + criterion(output_labels, target_labels)

        # Compute validation loss or other metrics

    # Adjust learning rate
    lr_scheduler.step()

    # Calculate epoch duration
    epoch_time = time.time() - start_time

    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Time: {epoch_time:.2f} seconds')

# Save the trained model
torch.save(model.state_dict(), 'ssd_OD.pth')