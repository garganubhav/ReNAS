import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


from stubs.pytorch.layers import ConvLayer, PoolingLayer

class Cutout(object):
  def __init__(self, length):
    self.length = length

  def __call__(self, img):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - self.length // 2, 0, h)
    y2 = np.clip(y + self.length // 2, 0, h)
    x1 = np.clip(x - self.length // 2, 0, w)
    x2 = np.clip(x + self.length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img

def _data_transforms_cifar10(parameters):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if "cutout" in parameters["hyper_parameters"].keys():
    train_transform.transforms.append(Cutout(parameters["hyper_parameters"]["cutout"]["size"]))

  test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, test_transform

def _data_transforms_Imagenet(parameters):
  IMAGENET_MEAN = [0.485, 0.456, 0.406]
  IMAGENET_STD = [0.229, 0.224, 0.225]

  train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
  ])
  if "cutout" in parameters["hyper_parameters"].keys():
    train_transform.transforms.append(Cutout(parameters["hyper_parameters"]["cutout"]["size"]))

  test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
  return train_transform, test_transform

def _data_transforms_cancer(parameters):
  train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

  test_transform = transforms.Compose([
    transforms.ToTensor()])

  return train_transform, test_transform

def get_train_test_queue(parameters):

  mode = parameters["trial_parameters"]["mode"]

  if parameters["trial_parameters"]["dataset"] == "cifar10":
    train_transform, test_transform = _data_transforms_cifar10(parameters)

    train_data = dset.CIFAR10(root=parameters["trial_parameters"]["data_dir"],
        train=True, download=True, transform=train_transform)

    if mode == "search_arch":
      train_portion = 0.8

      indices = list(range(len(train_data)))
      split = int(train_portion * len(train_data))

      train_queue = torch.utils.data.DataLoader(train_data,
        batch_size=parameters["hyper_parameters"]["batch_size"],
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4)

      valid_queue = torch.utils.data.DataLoader(train_data,
        batch_size=parameters["hyper_parameters"]["batch_size"],
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:len(train_data)]),
        pin_memory=True, num_workers=4)

    test_data = dset.CIFAR10(root=parameters["trial_parameters"]["data_dir"],
      train=False, download=True, transform=test_transform)
    
  
  elif parameters["trial_parameters"]["dataset"] == "Imagenet":
    train_transform, test_transform = _data_transforms_Imagenet(parameters)
    
    train_data = dset.ImageFolder(root=parameters["trial_parameters"]["data_imagenet_train"],
      transform=train_transform)

    test_data = dset.ImageFolder(root=parameters["trial_parameters"]["data_imagenet_test"],
      transform=test_transform)

  elif parameters["trial_parameters"]["dataset"] == "SVHN":
    pass

  elif parameters["trial_parameters"]["dataset"] == "cancer":
    train_transform, test_transform = _data_transforms_cancer(parameters)
    train_portion = 0.8
    # fix the transform later
    complete_data = dset.ImageFolder(root=parameters["trial_parameters"]["data_dir"], transform=test_transform)
    train_length = int(train_portion * len(complete_data))
    test_length = len(complete_data) - train_length

    train_data, test_data = torch.utils.data.random_split(complete_data, (train_length, test_length))

  test_queue = torch.utils.data.DataLoader(test_data,
      batch_size=parameters["hyper_parameters"]["batch_size"],
      shuffle=False, pin_memory=True, num_workers=4)

  if mode == "final_arch":
    train_queue = torch.utils.data.DataLoader(train_data,
      batch_size=parameters["hyper_parameters"]["batch_size"],
      shuffle=True, pin_memory=True, num_workers=4)
   
    return train_queue, test_queue

  else:
    return train_queue, valid_queue, test_queue


def get_init_model(init_model_description, in_channels):
  arg = {}
  layers = []

  for num_layer, layer in enumerate(init_model_description["layers"], 1):
    arg = {"name": "init_layer_{}_".format(num_layer)}
    arg["dropout"] = init_model_description["dropout"]
    if layer["block"] == "conv2d":
      arg["kernel_size"] = layer["kernel_size"]
      arg["in_channels"] = in_channels
      arg["out_channels"] = layer["outputs"]
      arg["stride"] = layer.get("stride") or 1
      arg["name"] = "init_"
      if "activation" in layer.keys():
        arg["activation"] = layer["activation"]
      layers.append(ConvLayer(arg))
    in_channels = arg["out_channels"]

  return layers, arg["out_channels"]
