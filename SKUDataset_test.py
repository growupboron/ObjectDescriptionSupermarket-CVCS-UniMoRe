
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datasets import SKUDataset
transform = transforms.Compose([transforms.ToPILImage()])
# Load the dataset
dataset = SKUDataset("train")

# Create a transform to display the images


# Iterate over the dataset and display the images
for i in range(len(dataset)):
    data = dataset[i]

    plt.imshow(data[0])
    plt.show()
