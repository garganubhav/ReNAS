import torch.nn as nn
from stubs.pytorch.layers import *

class CrossEntropyLabelSmooth(nn.Module):
  "Taken from darts gh repo"
  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

class AuxiliaryHeadCIFAR(nn.Module):
  "Taken from darts gh repo"
  def __init__(self, C, num_classes):
    """assuming input size 7x7"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(4, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

class AuxiliaryHeadImageNet(nn.Module):
  "Taken from darts gh repo"
  def __init__(self, C, num_classes):
    """assuming input size 14x14"""
    super(AuxiliaryHeadImageNet, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x


def get_final_model(final_model_description, in_channels):
  layers = []
  for layer in final_model_description["layers"]:
    arg = {"name": "final_layer_{}_".format(layer)}
    if layer["block"] == "avg_pool":
      # global average pooling
      layers.append(GlobalAvgPooling(arg))
    elif layer["block"] == "fc":
      # fully connected layer
      #arg["dropout"] = final_model_description["dropout"]
      arg["out_features"] = layer["outputs"]
      arg["num_features"] = in_channels
      layers.append(FullyConnectedLayer(arg))

  return layers
