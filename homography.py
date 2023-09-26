
import torch
import cv2
import numpy as np
from torchvision.models.detection import ssdlite320_mobilenet_v3_large



class HomographyTransform:

    def __init__(self, src_points, dst_points):
        self.src_points = np.array(src_points, dtype=np.float32)
        self.dst_points = np.array(dst_points, dtype=np.float32)
        self.matrix, _ = cv2.findHomography(self.src_points, self.dst_points)

    def apply_transform(self, points):
        points = np.array(points, dtype=np.float32)
        points = np.expand_dims(points, axis=1)
        transformed = cv2.perspectiveTransform(points, self.matrix)
        return transformed[0]

