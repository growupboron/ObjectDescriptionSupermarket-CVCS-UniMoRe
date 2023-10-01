import torch
import cv2
import numpy as np
import torch.nn as nn
from datasets import GroceryStoreDataset
from homography import HomographyTransform
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils
from functools import partial

# Constants
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor()
])

class ObjectCounter:
    def __init__(self, trained_model_path, classifier_model_path, homography_src_points, homography_dst_points):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models(trained_model_path, classifier_model_path)
        self.grocery_dataset = GroceryStoreDataset(split='test', transform=TEST_TRANSFORM)
        self.homography = HomographyTransform(homography_src_points, homography_dst_points)

    def _load_models(self, trained_model_path, classifier_model_path):
        """Load the SSD and classifier models."""
        # Load SSD model
        self.trained_model = ssdlite320_mobilenet_v3_large(pretrained=True)
        in_channels = det_utils.retrieve_out_channels(self.trained_model.backbone, (320, 320))
        num_anchors = self.trained_model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        dropout = nn.Dropout(p=0.5)
        self.trained_model.head.classification_head = nn.Sequential(
            SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer),
            dropout
        )
        checkpoint = torch.load(trained_model_path)
        self.trained_model.load_state_dict(checkpoint["model_state_dict"])
        self.trained_model.to(self.device).eval()

        # Load classifier model
        self.classifier_model = torchvision.models.densenet121(pretrained=True).to(self.device)
        self.classifier_model.classifier = nn.Linear(1024, 81).to(self.device)
        self.classifier_model.to(self.device).eval()

    def count_objects_and_relations(self, image_path):
        """Count objects, classify them, and identify spatial relationships."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image from path: {image_path}")

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            prediction = self.trained_model(image_tensor)
            bounding_boxes = prediction[0]['boxes'].cpu().numpy()

        rois = [image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :] for box in bounding_boxes]
        predictions = [self.grocery_dataset.classes[str(self._classify_roi(roi))] for roi in rois]
        positions_3d = self.homography.apply_transform(bounding_boxes[:, 2:])
        # Note: The identify_spatial_relationships function needs to be updated to handle orientations and bounding_boxes
        relationships = self._identify_spatial_relationships(positions_3d, None, bounding_boxes)

        return len(bounding_boxes), positions_3d, relationships, predictions

    def _classify_roi(self, roi):
        """Classify the Region of Interest (RoI) using the pre-trained classifier."""
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()])
        roi = transform(roi).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.classifier_model(roi)
            predicted_class = torch.argmax(output).cpu().item()
        return predicted_class

    def _identify_spatial_relationships(self, positions_3d, orientations, bounding_boxes, min_distance=10.0, max_angle_diff=30.0):
    """Identify spatial relationships between objects."""
    relationships = []

    # Calculate pairwise distances and angles between all objects
    num_objects = len(positions_3d)
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            distance = np.linalg.norm(positions_3d[i] - positions_3d[j])

            # Check spatial relationships based on distance and angle
            if distance < min_distance:
                angle_diff = self._calculate_angle_difference(orientations[i], orientations[j])
                if angle_diff <= max_angle_diff:
                    relationship = self._get_spatial_relationship(positions_3d[i], positions_3d[j], bounding_boxes[i], bounding_boxes[j])
                    relationships.append((f"Object {i+1}", relationship, f"Object {j+1}"))

    return relationships

def _calculate_angle_difference(self, angle1, angle2):
    """Calculate the absolute difference between two angles."""
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)

def _get_spatial_relationship(self, position1, position2, bbox1, bbox2):
    """Calculate spatial relationship based on angle and relative positions."""
    # Calculate center points of bounding boxes
    center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
    center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])

    # Calculate vector between the centers
    vector = center2 - center1

    # Calculate the angle between the vector and the line connecting the two objects
    angle = np.degrees(np.arctan2(vector[1], vector[0]))

    # Determine spatial relationship based on angle and relative positions
    if angle >= -45 and angle < 45:
        relationship = "to the right of"
    elif angle >= 45 and angle < 135:
        relationship = "above"
    elif angle >= -135 and angle < -45:
        relationship = "below"
    else:
        relationship = "to the left of"

    return relationship


if __name__ == "__main__":
    # Example usage
    counter = ObjectCounter("path_to_trained_model", "path_to_classifier_model", "src_points", "dst_points")
    num_objects, positions, relationships, predictions = counter.count_objects_and_relations("path_to_image")
    print(f"Detected {num_objects} objects.")
    print("Predictions:", predictions)
