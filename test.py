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
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import _utils as det_utils
from torch.nn import SmoothL1Loss
from torch.utils.tensorboard import SummaryWriter
from datasets import SKUDataset, TEST_TRANSFORM, TRAIN_TRANSFORM
import yaml
from functools import partial

import matplotlib.pyplot as plt

cudnn.benchmark = True

index_to_label = {
    0: 'object',
    1: 'background'
}



def visualize_results(images, outputs):
    for i in range(len(images)):
        image = images[i].permute(1, 2, 0).cpu().detach().numpy()
        boxes = outputs[i]['boxes'].cpu().detach().numpy()
        labels = outputs[i]['labels'].cpu().detach().numpy()
        scores = outputs[i]['scores'].cpu().detach().numpy()

        fig, ax = plt.subplots()
        ax.imshow(image)

        for box, label, score in zip(boxes, labels, scores):
            # print(label)
            if score > 0.5:
                x1, y1, x2, y2 = box
                print(f"box: {box}\tscore{score}")
                # Draw the bounding box on the image.
                ax.plot([x1, x2], [y1, y2], color='red', linewidth=2)
                # Write the label on the image.
                ax.text(x1, y1, index_to_label[label], color='red', fontsize=10)

        plt.savefig('./train_results/img_out_'+str(i)+'.jpeg')
        plt.close(fig)



def test(args, config):
    # Load config file
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Load model
    if args.model == 'ssd':
        model = ssdlite320_mobilenet_v3_large(pretrained=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

        model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer)
        checkpoint = torch.load('./checkpoints/checkpoint_ssd.pth')
    else:
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 2  # Object or background

        # Modify the model's output layer to match the number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Load weights from checkpoint file
        checkpoint = torch.load('./checkpoints/checkpoint_frcnn.pth')
    

    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    model.eval()

    # Load test images
    test_dataset = SKUDataset(split='val', transform=TEST_TRANSFORM)
    test_dataloader = DataLoader(test_dataset, batch_size=10, num_workers=2)

    # Test images
    for i, (images, x1, y1, x2, y2, class_id, image_width, image_height) in enumerate(test_dataloader):
        images = images.to(device)
        
        
        # Forward pass
        outputs = model(images)

        # Visualize results
        visualize_results(images, outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SKU Testing Script")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--model", type=str, default='ssd', help="type of model you want to use")
    parser.add_argument("--weigths", required=True, help="Path to the trained model")
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Test model
    test(args, config)
