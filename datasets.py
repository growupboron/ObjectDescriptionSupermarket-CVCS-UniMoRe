# create a custom dataset class for each dataset
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
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
    transforms.Normalize(mean=mean, std=std),  # normalize the image
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
        self.root = "Datasets/GroceryStoreDataset01"
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
