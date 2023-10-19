from datasets import SKUDataset
from count import ObjectCounter
from text_transformer import SceneDescriptionGenerator
import os
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

from datasets import GroceryStoreDataset

TEST_SIZE = 2940

TEST_TRANSFORM = transforms.Compose([

    transforms.Resize((256, 256)),  # resize the image to 256x256 pixels
    transforms.CenterCrop((224, 224)),

    transforms.ToTensor(),  # convert the image to a PyTorch tensor
    # transforms.Normalize(mean=mean, std=std)  # normalize the image
])

if __name__ == '__main__':
    trained_model_path = '/work/cvcs_2023_group23/ObjectDescriptionSupermarket-CVCS-UniMoRe/checkpoints/checkpoint.pth' 
    classifier_path = '/work/cvcs_2023_group23/ObjectDescriptionSupermarket-CVCS-UniMoRe/classifier.pth'
    
    grocery_dataset = GroceryStoreDataset(split='test', transform=TEST_TRANSFORM)
    
    index = np.random.uniform(0, TEST_SIZE)

    img_name = 'test_{}.jpg'.format(int(index))
    image_path = os.path.join('/work/cvcs_2023_group23/SKU110K_fixed/images', img_name)
    
    annot = pd.read_csv('/work/cvcs_2023_group23/SKU110K_fixed/annotations/annotations_test.csv')
    _, row = next(annot[annot['image_name'] == img_name].iterrows())
    print(row)
    width, height = row['image_width'], row['image_height']
    # Define source points for homography (in the order: top-left, top-right, bottom-right, bottom-left)
    homography_src_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Define destination points for homography (where you want the source points to be mapped to)

    homography_dst_points = np.array([[0, 0], [300, 0], [300, 200], [0, 200]], dtype=np.float32)

    
    
    object_counter = ObjectCounter(trained_model_path, classifier_path, homography_src_points, homography_dst_points)

    num_objects, positions_3d, relationships = object_counter.count_objects_and_relations(image_path)
    
    tokenizer = SceneDescriptionGenerator()
    
    description = tokenizer.generate_description(num_objects, positions_3d, relationships)
    
    print(f"Number of objects detected: {num_objects}")
    
    print(f"3D Positions: {positions_3d}")
    
    print(f"Final Scene description: {description}")
