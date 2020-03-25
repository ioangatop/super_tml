import torch.nn as nn
from torchvision import models


from src.utils import args


# ----- Model selection -----

def load_model(model_name=args.model):
    # Model selection
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 3)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, 3)

    model = nn.DataParallel(model.to(args.device))
    return model


if __name__ == "__main__":
    pass

