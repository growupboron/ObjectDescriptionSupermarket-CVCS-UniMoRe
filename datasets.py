# create a custom dataset class for each dataset
import ast
import json
import os

import numpy as np
from PIL import Image
import torch
from PIL import ImageDraw
from torch.utils.data import Dataset, DataLoader
import cv2

import torchvision
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# create a custom dataset class for each dataset
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),  # resize the image to 256x256 pixels
    transforms.CenterCrop(224),  # crop the image to 224x224 pixels around the center
    transforms.ToTensor(),  # convert the image to a PyTorch tensor
    # transforms.Normalize(mean=mean, std=std),  # normalize the image
    transforms.GaussianBlur(kernel_size=(5, 5)),
    transforms.RandomHorizontalFlip(p=0.5),  #
    # transforms.RandomRotation(degrees=180),  # data augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),

    # see https://pytorch.org/docs/stable/torchvision/transforms.html for more transforms
    #
])


class GroceryStoreDataset01(Dataset):
    # directory structure:
    # - root/[train/test/val]/[vegetable/fruit/packages]/[vegetables/fruit/packages]_class/[vegetables/fruit/packages]_subclass/[vegetables/fruit/packages]_image.jpg
    # - root/classes.csv
    # - root/train.txt
    # - root/test.txt
    # - root/val.txt
    def __init__(self, split='train', transform=None):
        super(GroceryStoreDataset01, self).__init__()
        self.root = "Datasets/GroceryStoreDataset-1/dataset"
        self.split = split
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}

        classes_file = os.path.join(self.root, "classes.csv")

        self.classes = {}
        with open(classes_file, "r") as f:

            lines = f.readlines()

            for line in lines[1:]:
                class_name, class_id, coarse_class_name, coarse_class_id, iconic_image_path, prod_description = line.strip().split(
                    ",")
                self.classes[class_id] = class_name
                self.class_to_idx[class_name] = coarse_class_id
                self.idx_to_class[class_id] = class_name
                self._description = prod_description

        self.samples = []
        split_file = os.path.join(self.root, self.split + ".txt")
        # print(self.classes)
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                img_path, class_id, coarse_class_id = line.split(",")
                class_name = self.classes[class_id.strip()]
                self.samples.append((os.path.join(self.root, img_path), int(self.class_to_idx[class_name])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        # print(img.shape, label)
        return img, label

    def description(self):
        return self._description


def collate_fn(batch):
    """
    Collate function that pads sequences to the same length.
    """
    print(batch[0][1])
    inputs = [torch.clone(item[0]).detach() for item in batch]
    targets = [torch.clone(item[1]).detach() for item in batch]

    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    # print(inputs_padded.shape, targets_padded.shape)
    return inputs_padded, targets_padded


class FreiburgDataset(Dataset):
    """class for loading the Freiburg dataset"""

    def __init__(self, split='train', num_split=0, transform=None):
        super(FreiburgDataset, self).__init__()
        self.root = "Datasets/freiburg_groceries_dataset/images"
        self.split = split
        self.num_split = num_split
        self.transform = transform
        self.samples = []
        self.classes = []
        self.labels = []
        self.classId_file = "Datasets/freiburg_groceries_dataset/classid.txt"
        self.load_classes()
        self.split_dir = os.path.join(self.root, self.split + str(self.num_split))
        self.load_samples()

    def load_classes(self):
        with open(self.classId_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                class_name, class_label = line.strip().split()
                self.classes.append(class_name)
                self.labels.append(class_label)

    def load_samples(self):
        with open(os.path.join(self.split_dir, ".txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                img_path, class_id = line.strip().split()
                self.samples.append((os.path.join(self.split_dir, img_path), int(class_id)))

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


class ShelvesDataset(Dataset):
    """class for loading the Shelves dataset for object detection"""

    # structure: root/[images/annotations]/[file_name].jpg/json

    def __init__(self, transform=None, max_num_boxes=10):
        super(ShelvesDataset, self).__init__()

        self.root = os.path.join("Datasets", "Supermarket+shelves", "Supermarket shelves", "Supermarket shelves")
        self.transform = transform
        self.num_files = len(os.listdir(os.path.join(self.root, "images")))
        self.max_num_boxes = max_num_boxes

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        torch.random.manual_seed(0)
        img_path = os.path.join(self.root, "images")
        img_filename = os.listdir(img_path)[idx]
        ret = [{'boxes': None, 'labels': [], 'image_id': torch.tensor([int(img_filename[:3])]), 'area': torch.tensor([4000 * 4000]),
                 'iscrowd': torch.tensor([[0, 0]])}]
        annotation_path = os.path.join(self.root, "annotations")
        annotation_filename = os.listdir(annotation_path)[idx]

        # read the image
        img = Image.open(os.path.join(img_path, img_filename)).convert('RGB')
        labels = []
        # Load the JSON annotation file
        with open(os.path.join(annotation_path, annotation_filename)) as f:
            annotatations = json.load(f)
        class_id_to_label = {}
        # Create an empty list to hold the padded bounding boxes
        padded_boxes = []
        # Iterate over the objects list
        for i, obj in enumerate(annotatations['objects']):
            # Extract the classId and the bounding box coordinates
            class_id = obj['classId']
            x1, y1 = obj['points']['exterior'][0]
            x2, y2 = obj['points']['exterior'][1]
            box = [x1, y1, x2, y2]
            padded_boxes.append(box)
            labels.append(0 if class_id == 10213293 else 1)

        # Pad the bounding boxes and labels to a fixed length
        padded_boxes = pad_boxes(padded_boxes, 100)
        labels = pad_labels(labels, 100)

        # return the image and the correspondent padded bounding boxes and labels
        if self.transform:
            img = self.transform(img)

        ret[0]['boxes'] = torch.tensor(padded_boxes)
        ret[0]['labels'] = torch.tensor(labels)
        return img, ret



def pad_boxes(boxes_list, pad_length):
    # pad the list with zeros if its length is less than the pad_length
    if len(boxes_list) < pad_length:
        boxes_list += [[0, 0, 1, 1]] * (pad_length - len(boxes_list))
    # truncate the list if its length is greater than the pad_length
    elif len(boxes_list) > pad_length:
        boxes_list = boxes_list[:pad_length]
    return boxes_list


def pad_labels(labels, max_num_boxes):
    # Add zeros to the labels list until it has a length of max_num_boxes
    while len(labels) < max_num_boxes:
        labels.append(0)
    return labels
