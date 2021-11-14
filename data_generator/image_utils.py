import numpy as np
import cv2

def image_loader(image_id, dataset_path):
    image_path = dataset_path + image_id + '.jpg'
    image = parse_image(image_path)
    image_normalized = normalization(image)
    return image_normalized

def parse_image(image_path):
    image_ = cv2.imread(image_path)
    image = image_.copy()
    return image

def normalization(image):
    image = image.astype(np.float32)
    image /= 255.0
    return image
