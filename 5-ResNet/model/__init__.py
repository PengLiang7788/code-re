from resnet import resnet18, resnet50, resnet34, resnet101, resnet152

model_dict = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet34': resnet34,
    'resnet101': resnet101,
    'resnet152': resnet152
}


def create_model(model_name, num_classes):
    return model_dict[model_name](num_classes=num_classes)