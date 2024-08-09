from model.MobileNetV1 import v1
from model.MobileNetV2 import v2
from model.MobileNetV3 import create_v3_large, create_v3_small

model_dict = {
    "mobilenet_v1": v1,
    "mobilenet_v2": v2,
    "mobilenet_v3_small": create_v3_small,
    "mobilenet_v3_large": create_v3_large
}


def create_model(model_name, num_classes=1000, width_mult=1.0):
    if model_name not in model_dict:
        raise Exception("model name is not available! only (MobileNetV1, MobileNetV2, MobileNetV3)")
    return model_dict[model_name](num_classes=num_classes, width_mult=width_mult)
