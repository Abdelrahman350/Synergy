from tensorflow.keras.applications import MobileNetV2

def create_MobileNetV2(input_shape, include_top=False, classes=62):
    base_model = MobileNetV2(input_shape=input_shape, include_top=include_top,\
         weights='imagenet', classes=classes)
    return base_model
    
