import torch
import torchvision

net = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
print(isinstance(net.anchor_generator, torch.nn.Module))
print(isinstance(net.backbone.features, torch.nn.Module))
