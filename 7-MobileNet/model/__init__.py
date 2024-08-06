from MobileNetV1 import v1

model_dict = {
    "MobileNetV1": v1
}


def create_model(model_name, num_classes=1000, width_mult=1.0):
    if model_name not in model_dict:
        raise Exception("model name is not available! only (MobileNetV1, MobileNetV2, MobileNetV3)")
    return model_dict[model_name](num_classes=num_classes, width_mult=width_mult)