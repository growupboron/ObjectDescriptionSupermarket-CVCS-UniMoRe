import os
import argparse
import logging
import time
import math
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils
import torchvision.ops as ops
from torch.nn import SmoothL1Loss
from torch.utils.tensorboard import SummaryWriter
from datasets import SKUDataset, TEST_TRANSFORM, TRAIN_TRANSFORM, VAL_TRANSFORM, custom_collate_fn
import yaml
from functools import partial

cudnn.benchmark = True

def setup_logging(log_dir, verbose):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO if verbose else logging.WARNING, format=log_format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)


def train(args, config, tensorboard_writer):
    # Data loading setup
    train_dataset = SKUDataset(split='train', transform=TRAIN_TRANSFORM)
    val_dataset = SKUDataset(split='val', transform=VAL_TRANSFORM)
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2, collate_fn=custom_collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size']//2, shuffle=True, num_workers=2, collate_fn=custom_collate_fn, pin_memory=True)
    
    # Model setup
    model = ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    # model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer)
    # model.to(device)
    
    dropout = nn.Dropout(p=0.5)
    model.head.classification_head = nn.Sequential(
    SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer),
    dropout
        )
    model.cuda(0)
    # model = torch.nn.DataParallel(model)

    # Use weight decay in the optimizer, which is another way to fight overfitting
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
        )

    # ReduceLROnPlateau scheduler decreases learning rate when a metric has stopped improving
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    labels_criterion = ops.sigmoid_focal_loss
    boxes_criterion = ops.generalized_box_iou_loss

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
        for batch_idx, (images, x1, y1, x2, y2, class_id, image_width, image_height) in enumerate(train_progress):
            
            # Move images and targets to the device
            images = images.to(device)
               
            
            x1 = x1.to(device)
            y1 = y1.to(device)
            x2 = x2.to(device)
            y2 = y2.to(device)
            class_id = (class_id - class_id.min()).to(device)
            image_width = image_width.to(device)
            image_height = image_height.to(device)
            
            print(images.shape)
            # Create a list of target dictionaries
            targets = []
            for i in range(len(images)):
                target = {}
                target['boxes'] = torch.ones((4, x1[i].shape[0]))
                target["boxes"][0] = x1[i]
                target["boxes"][1] = y1[i]
                target["boxes"][2] = x2[i]
                target["boxes"][3] = y2[i]
                target["boxes"] = target["boxes"].T
                target["boxes"] = target["boxes"].to(device)
                target['labels'] = class_id[i].to(device)
                target['image_width'] = image_width[i].to(device)
                target['image_height'] = image_height[i].to(device)
                targets.append(target)

        
                
            # logging.info(f'Train target shape: {targets[0]["boxes"].shape}')

            # Forward pass
            outputs = model(images, targets)

            print(f"train output: {outputs}")
            
            
            # Check if 'bbox_regression' and 'classification' keys exist in outputs dictionary
            if 'bbox_regression' in outputs and 'classification' in outputs:
                loss_output_boxes = outputs['bbox_regression']
                loss_output_labels = outputs['classification']
            else:
                logging.info("Error: Keys 'bbox_regression' and 'classification' not found in outputs dictionary.")
                raise KeyError("Keys 'bbox_regression' and 'classification' not found in outputs dictionary.")
            # print(f'Train output shape: {loss_output_boxes.shape} {loss_output_labels.shape}')
            # Extract tensors from targets list
            target_boxes = torch.cat([target['boxes'] for target in targets])
            target_labels = torch.cat([target['labels'] for target in targets])

            # loss = criterion(output_boxes, target_boxes) + criterion(output_labels, target_labels)
            
            # we prolly dont need to compute loss since it is computed inside the forward pass of the model
            loss = loss_output_boxes + loss_output_labels
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
        for batch_idx, (images, x1, y1, x2, y2, class_id, image_width, image_height) in enumerate(val_progress):
    
            # Move images and annotations to the device
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
                target['boxes'] = torch.ones((4, x1[i].shape[0]))
                target["boxes"][0] = x1[i]
                target["boxes"][1] = y1[i]
                target["boxes"][2] = x2[i]
                target["boxes"][3] = y2[i]
                target["boxes"] = target["boxes"].T
                target["boxes"] = target["boxes"].to(device)
                target['labels'] = class_id[i].to(device)
                target['image_width'] = image_width[i].to(device)
                target['image_height'] = image_height[i].to(device)
                targets.append(target)

            # Forward pass
            outputs = model(images)
            # print(outputs)

            # Check if 'boxes' and 'classification' keys exist in outputs dictionary
            if outputs and 'boxes' in outputs[0] and 'labels' in outputs[0]:
                output_boxes = torch.cat([output['boxes'] for output in outputs])
                output_labels = torch.cat([output['labels'] for output in outputs])
            else:
                print(outputs)
                logging.info("Error: Keys 'boxes' and 'labels' not found in outputs dictionary.")
                raise KeyError("Keys 'boxes' and 'labels' not found in outputs dictionary.")

            # Extract tensors from targets list
            target_boxes = torch.cat([target['boxes'] for target in targets])
            target_labels = torch.cat([target['labels'] for target in targets])
            # print(f"Validation shapes: {output_boxes.shape} {target_boxes.shape}")
            
            # Adjust tensors shapes if needed
            if target_boxes.shape[0] < output_boxes.shape[0]:
                # Extract the first target_boxes.shape[0] elements from output_boxes and output_labels
                output_boxes = output_boxes[:target_boxes.shape[0], :]
                output_labels = output_labels[:target_labels.shape[0]]
                
            elif target_boxes.shape[0] > output_boxes.shape[0]:
                # Extract the first output_boxes.shape[0] elements from target_boxes and target_labels
                target_boxes = target_boxes[:output_boxes.shape[0], :]
                target_labels = target_labels[:output_labels.shape[0]]

            
            print(f"Validation shapes matched: {output_boxes.shape} {target_boxes.shape}")
            output_labels = output_labels.float()  # Convert to float tensor
            target_labels = target_labels.float()
            loss = boxes_criterion(output_boxes, target_boxes, reduction='mean') + labels_criterion(output_labels, target_labels, reduction='mean')

            # Compute validation loss or other metrics
            total_val_loss += loss.item()
            
            val_loss = total_val_loss / (batch_idx + 1)
            tensorboard_writer.add_scalar('Loss/Validation', val_loss, epoch * len(val_dataloader) + batch_idx)
            val_progress.set_postfix({'Loss': val_loss})

        # Adjust learning rate
        lr_scheduler.step(metrics=val_loss)

        # Calculate epoch duration
        epoch_time = time.time() - start_time

        # Print epoch statistics
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_train_loss / len(train_dataloader):.4f}, '
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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
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
