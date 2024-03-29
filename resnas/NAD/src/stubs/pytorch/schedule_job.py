import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from stubs.pytorch.input_pipeline import get_train_test_queue, get_init_model
from stubs.pytorch.core_model import get_main_model
from stubs.pytorch.final_pipeline import *

from stubs.pytorch.spawn_job import spawn_job

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_channels_from_layers(layer, mode):
  if hasattr(layer, "block"):
    # in case of reduction block
    return layer.block.out_channels
  elif mode == "search_arch":
    return layer.blocks[0].out_channels
  else:
    out_channels = 0
    for block in layer.blocks:
      out_channels += block.out_channels
    return out_channels

class ResNetBackbone(nn.Module):
  """docstring for ResNetBackbone"""
  def __init__(self, layers, reduction_layers, mode, aux_head=None):
    super(ResNetBackbone, self).__init__()

    self.phase = "train"
    self.epoch = 0

    self.aux_head = aux_head
    if self.aux_head:
      self.aux_type = aux_head["aux_type"]
      self.aux_layer = aux_head["aux_layer"]

    self.layers = nn.ModuleList(layers)
    self.external_layers = [1, len(self.layers), len(self.layers)-1]
    self.reduction_layers = list(map(lambda x: x+1, reduction_layers))
    for layer_num, layer in enumerate(self.layers, 1):

      if layer_num not in self.external_layers and layer_num % 2 == 0:
        if layer_num + 2 >= len(self.layers) - 1:
          # have reached the end of skipconns for core model
          break
        input_channels = get_channels_from_layers(layer, mode)
        output_channels = get_channels_from_layers(self.layers[layer_num+1], mode)
        if input_channels < output_channels:
          residual_bottleneck = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1),
                                              nn.BatchNorm2d(output_channels))
          setattr(self, str(layer_num+2), residual_bottleneck)

      if aux_head and layer_num == self.aux_layer:
        aux_layer_channel = get_channels_from_layers(layer, mode)

    if aux_head:
      if self.aux_type == "cifar10":
        self.aux = AuxiliaryHeadCIFAR(aux_layer_channel, 10)
      elif self.aux_type == "imagenet":
        self.aux = AuxiliaryHeadImageNet(aux_layer_channel, 1001)
      else:
       print("Unknown aux_type")
       exit()
        
  def set_phase(self, phase):
    self.phase = phase

  def set_epoch(self, epoch):
    self.epoch = epoch

  def forward(self, x):
    epoch, phase = self.epoch, self.phase

    for layer_num, layer in enumerate(self.layers, 1):

      layer.set_epoch(epoch)
      layer.set_phase(phase)

      if layer_num % 2 == 1 and layer_num >= 5 and layer_num not in self.external_layers:
        if hasattr(self, str(layer_num-1)):
          last_residual = F.max_pool2d(last_residual, 2)
          last_residual = getattr(self, str(layer_num-1))(last_residual)
        #print(layer_num, x.shape, last_residual.shape)
        x = x + last_residual
        last_residual = x

      #print(x.size())
      x = layer((x))
      # it is not 2 but actually the #init_layers + 1
      if layer_num == 2:
        last_residual = x

      if self.aux_head and layer_num == self.aux_layer and phase != "test":
        #print("shape={} before aux".format(x.shape))
        aux_head = self.aux(x)

    if self.aux_head and phase != "test":
      return (x, aux_head)
    else:
      return x

class DenseNetBackbone(nn.Module):
  """docstring for Model"""
  def __init__(self, layers, reduction_layers, mode):
    super(DenseNetBackbone, self).__init__()
    self.layers = nn.ModuleList(layers)
    self.external_layers = [1, 2, 3, len(self.layers), len(self.layers) - 1]
    self.reduction_layers = list(map(lambda x: x+1, reduction_layers))
    #i = 0
    for layer_num, layer in enumerate(self.layers, 1):
      if layer_num not in self.external_layers and layer_num not in self.reduction_layers:
        in_channels = get_channels_from_layers(self.layers[layer_num - 1], mode)
        out_channels = in_channels
        for prev_layer_num in range(layer_num - 2, 1, -1):
          out_channels += get_channels_from_layers(self.layers[prev_layer_num-1], mode)
        setattr(self, str(layer_num), nn.Conv2d(out_channels, in_channels, 1))


  def forward(self, inp):
    x, _ = inp
    stored_activations = []
    for layer_num, layer in enumerate(self.layers, 1):
      #self.prev2prev = self.prev
      #self.prev = x
      if layer_num not in self.external_layers and layer_num not in self.reduction_layers:
        concat_layer_list = [x]
        image_dim = x.shape[2]
        for prev_layer_num in range(layer_num - 2, 1, -1):
          prev_layer = stored_activations[prev_layer_num-1]
          # Keep pooling till the image dimensions match
          while (prev_layer.shape[2] != image_dim):
            prev_layer = F.max_pool2d(prev_layer, 2)
          concat_layer_list.append(prev_layer)
        # concat along channel dimesions
        x = torch.cat(concat_layer_list, dim = 1)
        # Reduce channels by bottleneck layer
        x = getattr(self, str(layer_num))(x)
        #print("after concating at layer {}, channels = {}".format(layer_num, x.shape[1]))
      x, _ = layer((x, _))
      #print("after convolving at layer {}, channels = {}".format(layer_num, x.shape[1]))
      stored_activations.append(x)

    return (x, _)

def construct_model(model_specification, mode):
  init_model_description = model_specification["init_model"]
  init_model_layers, in_channels = get_init_model(init_model_description, in_channels=3)

  main_model_description = model_specification["core_model"]
  main_model_layers, in_channels = get_main_model(main_model_description, in_channels, mode)
  #main_model_layers = ResNet(BasicBlock, [2,2,2,2])

  final_model_description = model_specification["final_model"]
  final_model_layers = get_final_model(final_model_description, in_channels)

  layers = init_model_layers + main_model_layers + final_model_layers

  #model = nn.Sequential(*layers)
  reduction_layers = model_specification["core_model"]["reduction_layers"]

  aux_head = model_specification["core_model"].get("aux_head")
    
  model = ResNetBackbone(layers, reduction_layers, mode, aux_head)
  print(model)
  #print("number of trainable parameters are {}".format(count_parameters(model)))
  return model

def get_arch_parameters(model):
  arch_parameters = []
  for name, param in model.named_parameters():
    #print("In alpha", name,param)
    if 'AP_path_alpha' in name:
      #print("ALPHA",name,param)
      arch_parameters.append(param)
  #print("arch_params =", arch_parameters)
  return arch_parameters

def get_network_parameters(model):

  full_network_parameters = model.named_parameters()
  only_network_parameters = []
  #print("FULL NETWORK PARAMS=", full_network_parameters)
  for name, param in full_network_parameters:
    #print("Name=", name, "PARAM=", param)
    if "AP_path_alpha" and "AP_path_wb" not in name:
      only_network_parameters.append(param)
      #print("Name=", name, "PARAM=", param)
  #print("NETWORK PARAMS=", only_network_parameters)
  return only_network_parameters

def get_optimizer(model, hyperparameters, phase):

  optimizer_parameters = hyperparameters["optimizer"]
  network_parameters = get_network_parameters(model)

  if phase == "search_arch":
    arch_parameters = get_arch_parameters(model)
    # Hardcode arch optimizer as of now
    arch_optimizer = optim.Adam(arch_parameters, lr = 0.006, betas=(0,0.999))
    #arch_optimizer = optim.SGD(arch_parameters, lr=0.1)

  if optimizer_parameters["type"] == "momentum":
    regularization = hyperparameters["regularization"]
    weight_optimizer = optim.SGD(network_parameters, lr=optimizer_parameters["initial_lr"], momentum=optimizer_parameters["momentum"], nesterov=optimizer_parameters["nesterov"], weight_decay=regularization["value"])
  if optimizer_parameters["type"] == "adam":
    regularization = hyperparameters["regularization"]
    weight_optimizer = optim.Adam(network_parameters, lr=optimizer_parameters["initial_lr"], weight_decay=regularization["value"])

  if phase == "final_arch":
    return weight_optimizer
  print("TOTAL trainable parameters=", len(arch_parameters)+len(network_parameters))
  return weight_optimizer, arch_optimizer

def get_scheduler(optimizer, lr_scheduler_parameters, budget):
  if lr_scheduler_parameters["type"] == "cosine_decay":
    #T_max = lr_scheduler_parameters["T_max"]
    #min_lr = lr_scheduler_parameters["min"]
    #last_epoch = -1
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max,
    #  eta_min=min_lr, last_epoch=last_epoch)

    lr_max = lr_scheduler_parameters["max"]
    lr_min = lr_scheduler_parameters["min"]
    epochs_max_initial = lr_scheduler_parameters["epochs_max_initial"]
    epochs_max_mul = lr_scheduler_parameters["width_multiplier"]

    def cosineAnnealing(curr_epoch):
      
      def _get_lr(curr_epoch):
        curr_epoch += 1 #bcz pytorch starts with zero
        last_reset = 0
        mul = 1

        while(last_reset < curr_epoch):
          last_reset = epochs_max_initial * mul + last_reset
          mul *= epochs_max_mul

        if last_reset == curr_epoch:
          return _update(epochs_max_initial * mul, 1)
        else:
          mul /= epochs_max_mul
          last_reset -= epochs_max_initial * mul
          return _update(epochs_max_initial * mul, curr_epoch-last_reset)

      def _update(current_width, curr_epoch):
        print(current_width, curr_epoch)
        # Here curr_epoch refers to #epochs after last reset
        rate = torch.tensor((float(curr_epoch) / float(current_width)) * 3.1415926)
        #print("Curr_Epoch",float(curr_epoch),"curr_width=",float(current_width),"div=",(float(curr_epoch) / float(current_width)),"RATE=",rate,"COS=",torch.cos(rate))
        learning_rate = lr_min + (0.5 * (lr_max - lr_min) * (1.0 + torch.cos(rate)))
        print(learning_rate)
        return learning_rate

      return _get_lr(curr_epoch)
    

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, cosineAnnealing)

  elif lr_scheduler_parameters["type"] == "exponential_decay":
    gamma = lr_scheduler_parameters["gamma"]
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

  else:
    def linearLR(curr_epoch):
      beta = 1 - (float(curr_epoch)/float(budget))
      return beta
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, linearLR)

  return scheduler

def get_train_parameters(job_specification):
  parameters = {}
  parameters["trial_name"] = job_specification["trial_parameters"]["trial_name"]
  parameters["epochs"] = job_specification["trial_parameters"]["epochs"]
  parameters["save_frequency"] = job_specification["trial_parameters"]["save_frequency"]
  parameters["output_dir"] = job_specification["trial_parameters"]["output_path"]
  parameters["use_gpu"] = job_specification["system_parameters"]["gpus"]

  return parameters

def schedule_job(job_specification):

  log_dir = job_specification["trial_parameters"]["output_path"] + job_specification["trial_parameters"]["trial_name"]
  writer = SummaryWriter(log_dir=log_dir)

  use_gpu = job_specification["system_parameters"]["gpus"]
  hyper_parameters = job_specification["hyper_parameters"]
  components = {}

  mode = job_specification["trial_parameters"]["mode"]
  random_seed = job_specification["trial_parameters"]["seed"]
  model_specification = job_specification["model_specification"]
  model = construct_model(model_specification, mode)
  if "label_smoothing" in hyper_parameters.keys():
    criterion = CrossEntropyLabelSmooth(hyper_parameters["label_smoothing"]["num_classes"],
                  hyper_parameters["label_smoothing"]["value"])
  else:
    criterion = nn.CrossEntropyLoss()

  if use_gpu:
    #cudnn.benchmark = True
    model = nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()
    #os.environ['CUDA_VISIBLE_DEVICES'] ="1"
    #model=model.cuda()
    criterion = criterion.cuda()

    
  # Adds the model to tensorboard for visualization and debugging
  #writer.add_graph(model)
  #exit()
  components["parameters"] = get_train_parameters(job_specification)
  components["model"] = model
  components["criterion"] = criterion
  components["mode"] = mode
  components["writer"] = writer
  components["gradient_clipping"] = hyper_parameters.get("gradient_clipping")

  if "aux_head" in model_specification["core_model"].keys():
    components["aux_head"] = model_specification["core_model"]["aux_head"]
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  np.random.seed(random_seed)

  if mode == "search_arch":
    train_queue, valid_queue, test_queue = get_train_test_queue(job_specification)
    components["weight_optimizer"], components["arch_optimizer"] = get_optimizer(model, hyper_parameters, mode)
    components["train_queue"], components["valid_queue"], components["test_queue"] = train_queue, valid_queue, test_queue
  else:
    train_queue, test_queue = get_train_test_queue(job_specification)
    components["weight_optimizer"] = get_optimizer(model, hyper_parameters, mode)
    components["train_queue"], components["test_queue"] = train_queue, test_queue

  components["w_scheduler"] = get_scheduler(components["weight_optimizer"], hyper_parameters["lr_scheduler"], components["parameters"]["epochs"])
  #components["arch_scheduler"] = get_scheduler(components["arch_optimizer"], hyper_parameters["lr_scheduler"])

  spawn_job(components)

  writer.export_scalars_to_json(log_dir + "/scalars.json")
  writer.close()
