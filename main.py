from datasets import SKUDataset
from count import ObjectCounter

if __name__ == '__main__':
    trained_model_path = "path_to_trained_model.pth"
    homography_src_points = [...]  # Define source points for homography
    homography_dst_points = [...]  # Define destination points for homography
    image_path = "path_to_input_image.jpg"

    object_counter = ObjectCounter(trained_model_path, homography_src_points, homography_dst_points)

    num_objects, positions_3d, relationships = object_counter.count_objects_and_relations(image_path)
    print(f"Number of objects detected: {num_objects}")
    print(f"3D Positions: {positions_3d}")
    print("Spatial Relationships:")
    for rel in relationships:
        print(rel)
