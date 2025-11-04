import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Resnet(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super(Resnet, self).__init__()

        num_classes = self.cfg.model.num_classes
        pretrained_weights = self.cfg.model.pretrained_weights

        state = torch.load(pretrained_weights)
        self.backbone = resnet50(weights=None)
        self.backbone.load_state_dict(state)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
