import numpy as np
import cv2

def image_loader(image_id, dataset_path, input_shape):
    image_path = dataset_path + image_id + '.jpg'
    image = parse_image(image_path)
    image, aspect_ratio = resize_image(image, input_shape)
    image_normalized = normalization(image)
    return image_normalized, aspect_ratio

def parse_image(image_path):
    image_ = cv2.imread(image_path)
    image = image_.copy()
    return image

def normalization(image):
    image = image.astype(np.float32)
    image /= 255.0
    return image

def resize_image(image, input_shape=(224, 224)):
    original_shape = image.shape
    image = cv2.resize(image, input_shape[0:2], interpolation = cv2.INTER_AREA)
    resized_shape = image.shape
    aspect_0 = resized_shape[0]/float(original_shape[0])
    aspect_1 = resized_shape[1]/float(original_shape[1])
    aspect_ratio = (aspect_0, aspect_1)
    return image, aspect_ratio