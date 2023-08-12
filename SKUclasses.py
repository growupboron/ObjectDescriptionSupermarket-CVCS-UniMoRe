from datasets import SKUDataset

# Create an instance of the SKU110 dataset
dataset = SKUDataset(split='train', transform=None)  # Use the appropriate split

# Get all unique classes
unique_classes = set()
for image_id in dataset.ids:
    _, annotation = dataset.get_annotation(image_id)
    for obj in annotation['objects']:
        class_name = obj['class']
        unique_classes.add(class_name)

# Print the number of unique classes
num_classes = len(unique_classes)
print(f"Number of classes: {num_classes}")