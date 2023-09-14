import torch
import cv2
import numpy as np
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from homography import HomographyTransform

class ObjectCounter:
    def __init__(self, trained_model_path, homography_src_points, homography_dst_points, device="cuda"):
        self.device = device
        
        # Load trained model for object detection
        self.trained_model = ssdlite320_mobilenet_v3_large(pretrained=False)
        self.trained_model.load_state_dict(torch.load(trained_model_path, map_location=device))
        self.trained_model.to(device)
        self.trained_model.eval()

        # Create HomographyTransform instance
        self.homography = HomographyTransform(homography_src_points, homography_dst_points)

    def count_objects_and_relations(self, image_path):
        image = cv2.imread(image_path)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            prediction = self.trained_model(image_tensor)
            bounding_boxes = prediction[0]['boxes'].cpu().numpy()

            # Apply homography to obtain 3D positions
            positions_3d = [self.homography.apply_transform(box[:2]) for box in bounding_boxes]

            # Identify spatial relationships
            relationships = self.identify_spatial_relationships(positions_3d)

        return len(bounding_boxes), positions_3d, relationships



    def identify_spatial_relationships(self, positions_3d, orientations, bounding_boxes, min_distance=10.0, max_angle_diff=30.0):
        relationships = []
        
        # Calculate pairwise distances and angles between all objects
        num_objects = len(positions_3d)
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                distance = np.linalg.norm(positions_3d[i] - positions_3d[j])
                
                # Check spatial relationships based on distance and angle
                if distance < min_distance:
                    angle_diff = calculate_angle_difference(orientations[i], orientations[j])
                    if angle_diff <= max_angle_diff:
                        relationship = get_spatial_relationship(positions_3d[i], positions_3d[j], bounding_boxes[i], bounding_boxes[j])
                        relationships.append((f"Object {i+1}", relationship, f"Object {j+1}"))
                    
        return relationships

def calculate_angle_difference(angle1, angle2):
    # Calculate the absolute difference between two angles
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)

def get_spatial_relationship(position1, position2, bbox1, bbox2):
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
