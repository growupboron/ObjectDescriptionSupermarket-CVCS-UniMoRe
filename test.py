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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.ops as ops
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import _utils as det_utils
from torch.nn import SmoothL1Loss
from torch.utils.tensorboard import SummaryWriter
from datasets import SKUDataset, TEST_TRANSFORM, TRAIN_TRANSFORM, custom_collate_fn
import yaml
from functools import partial
import torchvision.ops as ops
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torcheval.metrics.functional import binary_precision, binary_recall

cudnn.benchmark = False

index_to_label = {
    0: 'background',
    1: 'object'
}



def visualize_results(images, outputs, writer):
    out_imgs = []
    for i in range(len(images)):
        image = images[i].permute(1, 2, 0).cpu().detach().numpy()
        boxes = outputs[i]['boxes'].cpu().detach().numpy()
        labels = outputs[i]['labels'].cpu().detach().numpy()
        scores = outputs[i]['scores'].cpu().detach().numpy()

        fig, ax = plt.subplots()
        ax.imshow(image)
        n = 0
        for box, label, score in zip(boxes, labels, scores):
            n += 1
            # print(label)
            if score > 0.5 and n < 25:
                x1, y1, x2, y2 = box
                print(f"box: {box}\tscore{score}")
                # Draw the bounding box on the image.
                rect = patches.Rectangle((x1, y1), (x2-x1), (y2-y1), linewidth=1, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)
                # Write the label on the image.
                ax.text(x1, y1, index_to_label[label], color='red', fontsize=10)

        
        # Save the image to TensorBoard
        plt.savefig('./train_results/img_out_'+str(i)+'.jpeg')
        plt.close(fig)

        # Add the image to the TensorBoard writer
        img_tensor = torch.from_numpy(image).permute(2, 0, 1)
        out_imgs.append(img_tensor)
        #writer.add_image('image', img_tensor, i)
    




def test(args, config):
    # Load config file
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Initialize Tensorboard writer
    writer = SummaryWriter("./logs/")

    # Load model
    if args.model == 'ssd':
        model = ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

        model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer)
        checkpoint = torch.load('./checkpoints/checkpoint.pth')
    else:
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 2  # Object or background

        # Modify the model's output layer to match the number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load('./checkpoints/checkpoint.pth')
        model.load_state_dict(checkpoint["model_state_dict"])
        
    model.cuda(0)
    # model = torch.nn.DataParallel(model)
    model.eval()
    
    

    # Load test images
    test_dataset = SKUDataset(split='test', transform=TEST_TRANSFORM)
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=True, collate_fn=custom_collate_fn, num_workers=2, pin_memory=True)

    

    # Test images
    # Define the statistics you want to save
    statistics = {
        'num_ground_truth_boxes': [],
        'num_predicted_boxes': [],
        'iou': []
    }

    # Save the statistics to TensorBoard
    for i, (images, x1, y1, x2, y2, class_id, image_width, image_height) in enumerate(test_dataloader):
        
        images = images.to(device)
        x1 = x1.to(device)
        y1 = y1.to(device)
        x2 = x2.to(device)
        y2 = y2.to(device)
        class_id = (class_id - class_id.min()).to(device)
        image_width = image_width.to(device)
        image_height = image_height.to(device)
        
        

        # Forward pass
        
        outputs = model(images)
        for j in range(len(images)):
            outputs[j]['boxes'] = outputs[j]['boxes']

            # Calculate the statistics
            num_ground_truth_boxes = len(x1)
            num_predicted_boxes = outputs[j]['boxes'].size(0)
            bboxes = torch.ones((4, x1[j].shape[0]))
            bboxes[0] = x1[j]
            bboxes[1] = y1[j]
            bboxes[2] = x2[j]
            bboxes[3] = y2[j]
            bboxes = bboxes.T
            iou = ops.box_iou(outputs[j]['boxes'].cpu(), bboxes).mean()
            print(f'BBoxes IoU: {iou}')
            # precision = binary_precision(class_id[j], outputs[j]['labels'][:len(class_id[j])].cpu())
            # recall = binary_recall(class_id[j], outputs[j]['labels'][:len(class_id[j])].cpu())

            '''# Save the statistics
            statistics['num_ground_truth_boxes'].append(num_ground_truth_boxes)
            statistics['num_predicted_boxes'].append(num_predicted_boxes)
            statistics['iou'].append(iou)'''
          
        visualize_results(images, outputs, writer)


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

        

 