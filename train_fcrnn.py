import os
import argparse
import logging
import time
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
from torch.nn import SmoothL1Loss
from torch.utils.tensorboard import SummaryWriter
from datasets import SKUDataset, TEST_TRANSFORM, TRAIN_TRANSFORM
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import yaml
from functools import partial

cudnn.benchmark = False


def setup_logging(log_dir, verbose):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO if verbose else logging.WARNING, format=log_format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)



def train(args, config, tensorboard_writer):
    # Data loading setup
    train_dataset = SKUDataset(split='train', transform=TRAIN_TRANSFORM)
    val_dataset = SKUDataset(split='val', transform=TEST_TRANSFORM)
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'] // 2, shuffle=True, num_workers=2)
    
    # Model setup
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # Object or background

    # Modify the model's output layer to match the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = SmoothL1Loss()

    # Load checkpoint if available
    start_epoch = 0
    start_batch_idx = 0
    total_train_loss = 0.0
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        start_batch_idx = checkpoint['batch_idx'] + 1
        total_train_loss = checkpoint['total_train_loss']
        logging.info(f"Resuming training from epoch {start_epoch}, batch {start_batch_idx}")
        
    num_epochs = config['training']['num_epochs']
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        start_time = time.time()  # Start time for epoch
        train_progress = tqdm(train_dataloader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=True)
        total_train_loss = 0.0
        for batch_idx, (images, x1, y1, x2, y2, class_ids, image_widths, image_heights) in enumerate(train_progress):
            # Move images and targets to the device
            images = list(image.to(device) for image in images)
            x1 = x1.to(device)
            y1 = y1.to(device)
            x2 = x2.to(device)
            y2 = y2.to(device)
            class_ids = class_ids.to(device)
            image_widths = image_widths.to(device)
            image_heights = image_heights.to(device)
            
            # Create a list of target dictionaries
            targets = []
            for i in range(len(images)):
                target = {}
                target["boxes"] = torch.stack((x1[i], y1[i], x2[i], y2[i]), dim=1).to(device)
                target["labels"] = class_ids[i]
                target["image_width"] = image_widths[i]
                target["image_height"] = image_heights[i]
                targets.append(target)
                
            # Forward pass
            loss_list = model(images, targets)
            
            
            loss = sum(l for l in loss_list.values())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_loss = total_train_loss / (batch_idx + 1)
            tensorboard_writer.add_scalar('Loss/Train', train_loss, epoch * len(train_dataloader) + batch_idx)
            train_progress.set_postfix({'Loss': train_loss})

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_progress = tqdm(val_dataloader, desc=f'Validation [{epoch+1}/{num_epochs}]', leave=True)
        for batch_idx, (images, x1, y1, x2, y2, class_ids, image_widths, image_heights) in enumerate(val_progress):
            # Move images and targets to the device
            images = list(image.to(device) for image in images)
            x1 = x1.to(device)
            y1 = y1.to(device)
            x2 = x2.to(device)
            y2 = y2.to(device)
            class_ids = class_ids.to(device)
            image_widths = image_widths.to(device)
            image_heights = image_heights.to(device)
            
            # Create a list of target dictionaries
            targets = []
            for i in range(len(images)):
                target = {}
                target["boxes"] = torch.stack((x1[i], y1[i], x2[i], y2[i]), dim=1).to(device)
                target["labels"] = class_ids[i]
                target["image_width"] = image_widths[i]
                target["image_height"] = image_heights[i]
                targets.append(target)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(images)
            
            # Calculate loss
            model.train()
            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())
            model.eval()
            
            total_val_loss += loss.item()
            
            val_loss = total_val_loss / (batch_idx + 1)
            tensorboard_writer.add_scalar('Loss/Validation', val_loss, epoch * len(val_dataloader) + batch_idx)
            val_progress.set_postfix({'Loss': val_loss})

        # Adjust learning rate
        lr_scheduler.step()

        # Calculate epoch duration
        epoch_time = time.time() - start_time

        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_train_loss / len(train_dataloader):.4f}, '
            f'Validation Loss: {total_val_loss / len(val_dataloader):.4f}, Time: {epoch_time:.2f} seconds')

        # Save checkpoint after every epoch
        checkpoint_path = os.path.join("./checkpoints", 'checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'batch_idx': len(train_dataloader) - 1,  # Last batch index
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_train_loss': total_train_loss
        }, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="SKU Training Script")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--resume_checkpoint", help="Path to checkpoint file to resume training")
    parser.add_argument("--log_dir", required=True, help="Path to the log directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")  # Adjust batch size according to your GPU memory
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Overwrite config with CLI arguments
    config['logging']['log_dir'] = args.log_dir
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.learning_rate
    config['training']['num_epochs'] = args.num_epochs

    # Setup logging
    setup_logging(config['logging']['log_dir'], args.verbose)

    # Tensorboard setup
    tensorboard_writer = SummaryWriter(config['logging']['log_dir'])

    # Training
    train(args, config, tensorboard_writer=tensorboard_writer)


if __name__ == '__main__':
    main()
