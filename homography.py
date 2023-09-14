import cv2
import numpy as np


class HomographyTransform:

    def __init__(self, src_points, dst_points):
        self.src_points = np.array(src_points, dtype=np.float32)
        self.dst_points = np.array(dst_points, dtype=np.float32)
        self.matrix, _ = cv2.findHomography(self.src_points, self.dst_points)

    def apply_transform(self, points):
        transformed = cv2.perspectiveTransform(np.array([points], dtype=np.float32), self.matrix)
        return transformed[0]
