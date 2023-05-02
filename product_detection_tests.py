import torch
import torchvision
from PIL import Image
from PIL import ImageDraw
from torch import optim, nn, device
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import Compose
from torchvision.transforms import Resize, ToTensor, Normalize
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from datasets import ShelvesDataset

# Define the image preprocessing transforms
preprocess = Compose([
    Resize(size=(2000, 2000)),
    ToTensor(),
    #Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load the dataset
dataset = ShelvesDataset(transform=preprocess)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

# Load the object detection and segmentation model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)

#model.roi_heads = torch.nn.Linear(in_features=1024, out_features=2, bias=True)


# Load the 3D spatial reasoning model (not implemented in this code)
print("Loading image...",end='')
# Load an image
img = Image.open("./Datasets/Supermarket+shelves/Supermarket shelves/Supermarket shelves/images/004.jpg")
print("Done!")
# Apply the image preprocessing transforms
img_tensor = preprocess(img)
print("Detecting objects...",end='')
# img.show()
# Use the object detection and segmentation model to detect and segment objects
model.eval()
with torch.no_grad():
    prediction = model([img_tensor])[0]

# Extract the object labels and bounding boxes from the prediction
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
boxes = prediction["boxes"].tolist()
print("Done!")
# Extract the object regions from the image
regions = []
for box in boxes:
    x1, y1, x2, y2 = box
    region = img.crop((x1, y1, x2, y2))
    #region.show()
    regions.append(region)

# Use the 3D spatial reasoning model to reason about the layout of the objects

# Generate a natural language description of the scene
description = f"There are {len(regions)} objects in the scene. "
for i, region in enumerate(regions):
    description += f"The {labels[i]} is located at ({int(boxes[i][0])}, {int(boxes[i][1])}) with width {int(boxes[i][2] - boxes[i][0])} and height {int(boxes[i][3] - boxes[i][1])}. "
print(description)

# Visualize the object detection and segmentation results
# Convert boxes to a PyTorch tensor
boxes = torch.tensor(boxes)

# Convert boxes to (x1, y1, x2, y2) format
boxes = torchvision.ops.box_convert(boxes, in_fmt='xyxy', out_fmt='xywh')
boxes[:, 2:] += boxes[:, :2]
print(f'boxes: {boxes}')
# Visualize the object detection and segmentation results
'''masks = draw_bounding_boxes(img_tensor.type(torch.uint8), boxes=boxes,
                                 labels=labels,
                                 fill=True,
                                 colors="red",
                                 width=4, font_size=30)'''


img = to_pil_image(img_tensor.detach())

Draw = ImageDraw.Draw(img)
for box in boxes:

    Draw.rectangle(tuple(box), outline='red', width=4)
#img.show()

