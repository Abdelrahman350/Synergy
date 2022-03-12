from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

def create_backbone(input_shape, backbone='mobileNetV2', weights='imagenet', pooling=None):
     if backbone == 'mobileNetV2':
          backbone = MobileNetV2(input_shape=input_shape, include_top=False,\
               weights=weights, pooling=pooling)
     elif backbone == 'efficientNet':
          backbone = EfficientNetB0(input_shape=input_shape, include_top=False,\
               weights=weights, pooling=pooling)
     else:
          backbone = None
          print("This backbone is not implemented.")
     return backbone