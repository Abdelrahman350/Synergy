from turtle import back
from tensorflow.keras.applications import MobileNetV2

def create_MobileNetV2(input_shape, include_top=False, classes=62):
     backbone = MobileNetV2(input_shape=input_shape, include_top=include_top, weights='imagenet')
     return backbone