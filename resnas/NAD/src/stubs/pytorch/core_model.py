import torch
import torch.optim as optim
from stubs.pytorch.layers import *
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class SingleLayer(BasicUnit):
  def __init__(self, blocks, algorithm="nad", beta=None, reduction=False):
    super(SingleLayer, self).__init__()
    self.reduction = reduction
    self.algorithm = algorithm

    if not self.reduction and self.algorithm == "final_arch":
      self.block = blocks[0]
      return

    self.blocks = nn.ModuleList(blocks)
    self.AP_path_alpha = Parameter(torch.ones(len(self.blocks)))  # architecture parameters
    self.AP_path_wb = Parameter(torch.Tensor(len(self.blocks)))  # binary gates
    self.grads = []
    #self.loss = torch.zeros(size=(len(blocks),)).cuda()
    if self.algorithm == "plnas" and reduction == False:
      self.indexes = np.sort(np.random.choice(range(len(self.blocks)),
          size=(2), replace=False))
      self.prev_epochs = 0

    elif self.algorithm == "nad" and reduction == False:
      self.prev_epochs = torch.tensor(1)
      self.beta = torch.tensor(beta.item())
      self.policy_p1 = torch.tensor(1.0)
      self.sampled_freq = torch.zeros(size=(len(blocks),)).cuda()
      self.sampled_freq = torch.clamp(self.sampled_freq, min=1e-20)
      self.n_runs=torch.tensor(1.)
      self.c = 2.0
      self.lambda_ = torch.tensor(0.1)

    elif self.algorithm == "final_arch" and reduction == False:
      self.block = self.blocks[0]

  def pad_image(self, x, padding_type, kernel_size, stride=1, dilation=1):
    if padding_type == 'SAME':
        p0 = ((x.shape[2] - 1) * (stride - 1) + dilation * (kernel_size[0] - 1))# //2
        p1 = ((x.shape[3] - 1) * (stride - 1) + dilation * (kernel_size[1] - 1))# //2
        #print(x.shape, kernel_size, p0, p1)
        input_rows = x.shape[2]
        filter_rows = kernel_size[0]

        x = F.pad(x, [0, p1, 0, p0])
        return x

  def pad_image_v2(self, x, padding_type, kernel_size, stride=1, dilation=1):
    if padding_type == 'SAME':
      effective_kernel_size_0 = (kernel_size[0] - 1) * dilation + 1
      out_dim = (x.shape[2] + stride - 1) // stride
      p0 = max(0, (out_dim - 1) * stride + effective_kernel_size_0 - x.shape[2])

      effective_kernel_size_1 = (kernel_size[1] - 1) * dilation + 1
      out_dim = (x.shape[3] + stride - 1) // stride
      p1 = max(0, (out_dim - 1) * stride + effective_kernel_size_1 - x.shape[3])

      padding_before_0 = p0 // 2
      padding_before_1 = p1 // 2

      padding_after_0 = p0 - padding_before_0
      padding_after_1 = p1 - padding_before_1

      x = F.pad(x, [padding_before_1, padding_after_1, padding_before_0, padding_after_0])
      return x

  def get_indexes_random(self):
    indexes = np.sort(np.random.choice(range(len(self.blocks)),
          size=(2), replace=False))
    return indexes

  def get_indexes_darts(self):
    pass

  def get_indexes_plnas(self):
    self.epoch += 1
    probs = []
    for block in self.blocks:
      probs.append(block.alpha)
    probs = torch.stack(probs)
    sm_probs = F.softmax(probs, dim=0)
    self.indexes = indexes = torch.sort(torch.multinomial(sm_probs, 2, replacement=False))[0]
    return self.indexes

  def get_indexes_snas(self):
    pass

  def get_indexes_nad(self):
    """ Write this function's explanation here """

    #if self.phase == "test":
     # probs = F.softmax(self.AP_path_alpha / self.lambda_, dim=0)
      #indexes = torch.multinomial(probs.data, 2, replacement=False)
      #torch.gather might be the speed bottleneck here
      #todo compare perf. with probs[indexes[0/1]]
      #probs_slice = F.softmax(torch.stack([
       #         self.AP_path_alpha[idx] for idx in indexes
        #    ]), dim=0)
      #c = torch.multinomial(probs_slice.data, 1)[0]
      #return indexes, None
    #if self.prev_epochs == self.epoch:
     # self.policy_p1 *= self.beta
      #self.prev_epochs+=1
      #self.c = self.policy_p1.item()
    #print("EPOCH=", self.epoch, "PREV_EPOCH=", self.prev_epochs)
    #if self.prev_epochs <= self.epoch:
     # self.policy_p1 *= self.beta
     # print("Beta= ", self.beta)
     # self.prev_epochs += 1
    #print("POLICY P1=", self.policy_p1)
   # policy = torch.bernoulli(self.policy_p1).item()
   # print("POLICY=", policy)
    #if policy == 0:
      # Exploration policy
     # probs = F.softmax(self.AP_path_alpha.data/2 + self.c*(torch.sqrt(torch.log(self.n_runs)/self.sampled_freq)), dim=0)
    #else:
      #probs = F.softmax(-1 * self.sampled_freq, dim=0)
      #probs = torch.clamp(probs, min=0.01, max=0.96)
      #print(probs)
      #if self.n_runs < len(self.AP_path_alpha) :
        #index = self.n_runs
      #else:
      #probs = F.softmax(self.AP_path_alpha.data + (torch.sqrt(2*torch.log(float(self.n_runs+1)/self.sampled_freq))),dim=0)
      #indexes = torch.multinomial(probs.data, 2, replacement=False)
      #print("ALPHA BEFORE UCB =", self.AP_path_alpha.data)
      #print("FREQUENCIES",self.sampled_freq)
      #print("NRUNS=", self.n_runs+1)
      #print("DIVIDEND", float(self.n_runs+1)/self.sampled_freq)
      #print("LOG DIVIDEND", torch.log(float(self.n_runs+1)/self.sampled_freq))
      #print("CONFIDENCE BOUND=", (torch.sqrt(2*torch.log(float(self.n_runs+1)/self.sampled_freq))))
      #print("CB SIZE", (torch.sqrt(2*torch.log(float(self.n_runs+1)/self.sampled_freq))).size())
      #print("SORTED VALUES=", values)
      #print("INDEXES=", indexes)
      #index = indexes[0]
      #print("INDEX=", index)
      #indexes = torch.sort(torch.multinomial(probs, 2, replacement=False))[0]
      #new_probs = F.softmax(torch.gather(sm_self.AP_path_alpha, 0, indexes), dim=0)
      #if phase == "train":
        # If the phase is search_arch then we should not increase the sampling freq
      #index = indexes[0]
      #self.sampled_freq[indexes] += 1
     # self.n_runs += 1
    #else:
      # exploitation policy
    probs = F.softmax(self.AP_path_alpha/self.lambda_, dim=0)
    probs = torch.clamp(probs.data, min=1e-20)
    #print("PROBS IN WEIGHT TRAIN =",probs) 
    indexes = torch.multinomial(probs, 2, replacement=False).data
      #probs_slice = F.softmax(torch.stack([
                #self.AP_path_alpha[idx] for idx in indexes
            #]), dim=0)
      #print("PROBS SLICE= ",probs_slice)
      #index = torch.multinomial(probs_slice.data, 1)[0]
      # new_probs = F.softmax(torch.gather(probs, 0, indexes), dim=0, )
      #if phase == "train":
    #print("INDEXES AFTER EXPLOIT", indexes) #"ALPHAS=", self.AP_path_alpha[indexes])
      #index = indexes[0]
      #self.sampled_freq[index] += 1
    return indexes, None

  def apply_reduction(self, x):
    outputs = []

    for block in self.blocks:
      outputs.append(block(x))
    concat_output = torch.cat(outputs, dim=1)

    return concat_output

  def reset_binary_gates(self):
    #reset all gates
    self.grads = []
    self.AP_path_wb.data.zero_()
    if self.prev_epochs == self.epoch:
      self.policy_p1 *= self.beta
      print("SELF.POLICY_P1=",self.policy_p1) 
      self.prev_epochs+=1
    policy = torch.bernoulli(self.policy_p1).item()
    print("POLICY=", policy)
    print("ALPHAS=", self.AP_path_alpha.data)
    if  policy==1:
      probs = F.softmax(self.AP_path_alpha.data + self.c*(torch.sqrt(torch.log(self.n_runs)/self.sampled_freq)), dim=0)
      print("PROBS=", probs)
    else:
      probs = F.softmax(self.AP_path_alpha/self.lambda_, dim=0)
    probs = torch.clamp(probs.data, min=1e-20)
    #print("ALPHAS=", self.AP_path_alpha.data)
    sample_op = torch.multinomial(probs.data, 2, replacement=False)
    if policy==1:
      probs_slice = F.softmax(torch.stack([
              self.sampled_freq[idx] for idx in sample_op
          ]), dim=0)
      self.n_runs += 1
      self.sampled_freq[sample_op] += 1
      print("FREQUENCIES=", self.sampled_freq)
    else:
      probs_slice = F.softmax(torch.stack([
              self.AP_path_alpha[idx] for idx in sample_op
          ]), dim=0)
    c = torch.multinomial(probs_slice.data, 1)[0]
    active_op = sample_op[c].item()
    inactive_op = sample_op[1 - c].item()
    #else:
     # 
      #probs, indexes = F.softmax(self.AP_path_alpha.data/4 + self.c*(torch.sqrt(torch.log(self.n_runs)/self.sampled_freq)), dim=0)
      #print("PROBS=", probs, "INDEXES=", indexes)
      #print("ALPHAS=", self.AP_path_alpha.data/4)
      #probs_slice = torch.stack([probs[idx] for idx in range(2)]) #, dim=0) #F.softmax(self.AP_path_alpha / self.lambda_, dim=0)
      #index = torch.multinomial(probs_slice, 1)
      #print("PROBS SLICE=", probs_slice)
      #print("ACTIVE INDEX=", index)
      #active_op = indexes[index].item()
      #inactive_op = indexes[1-index].item()
      #print("ACTIVE PATH=", active_op, "INACTIVE_PATH=", inactive_op)
    self.active_index = [active_op]
    self.inactive_index = [inactive_op]
    self.AP_path_wb.data[active_op] = 1.0
    self.AP_path_wb.register_hook(lambda grad:self.grads.append(grad.clone()))
    #print("GRADS OF ALL PATHS=", self.grads)

  def forward(self, x):
    if self.reduction:
      output = self.apply_reduction(x)
      return output

    else:
      if self.algorithm == "final_arch":
        #print("INPUT SIZE=", x.size())
        k = self.block.kernel_size
        x = self.pad_image_v2(x, padding_type='SAME', kernel_size=k)
        x = self.block(x)
        return x

      elif self.algorithm == "plnas":
        # ProxylessNAS sampling
        indexes = self.get_indexes_plnas()
        new_probs = F.softmax(torch.gather(probs, 0, indexes), dim=0)

      elif self.algorithm == "nad" and self.phase is not 'validation':
        # Neural Arch. Design sampling
        indexes, new_probs = self.get_indexes_nad()

      elif self.algorithm == "snas":
        indexes = self.get_indexes_snas()

      elif self.algorithm == "darts":
        indexes = self.get_indexes_darts()

      elif self.algorithm == "random":
        # Random sampling
        indexes = self.get_indexes_random()

      output = 0

      if self.phase == "validation":
        #self.reset_binary_gates()
        a_i = self.active_index[0]
        in_i= self.inactive_index[0]
        k0 = self.blocks[a_i].kernel_size
        k1 = self.blocks[in_i].kernel_size
        x0 = self.pad_image(x, padding_type='SAME', kernel_size=k0)
        x1 = self.pad_image(x, padding_type='SAME', kernel_size=k1)
        oi = self.blocks[a_i](x0)
        output = output + (self.AP_path_wb[a_i]*oi)
        oi = self.blocks[in_i](x1)
        output = output + (self.AP_path_wb[in_i] * oi.detach())
	      
      else:
        #freq, indexes = torch.sort(self.sampled_freq, dim=0, descending=True)
        #print("MAX FRQ=", freq[0], "PATH=", indexes[0])
        k0 = self.blocks[indexes[0]].kernel_size
        k1 = self.blocks[indexes[1]].kernel_size
        x0 = self.pad_image(x, padding_type='SAME', kernel_size=k0)
        x1 = self.pad_image(x, padding_type='SAME', kernel_size=k1)
        output = output + (self.blocks[indexes[0]](x0))
        output += self.blocks[indexes[1]](x1)

    # merge by summation instead of concat operation
    #concat_output = torch.cat(output, dim=1)
    #self.AP_path_wb.register_hook(lambda grad:self.grads.append(grad.clone()))
    #print("GRADS OF ALL PATHS= ", self.grads)
    return output

  def unused_modules_off(self):
    self.unused = {}
    involved_index = self.active_index + self.inactive_index
    for i in range(len(self.blocks)):
      if i not in involved_index:
        self.unused[i] = self.blocks[i]
        self.blocks[i] = None

  def unused_modules_back(self):
    if self.unused is None:
      return
    for i in self.unused:
      self.blocks[i] = self.unused[i]
    self.unused = None

  def set_arch_param_grad(self):
    #print(self.grads)
    binary_grads = self.grads[0]
    #print(binary_grads)
    if self.AP_path_alpha.grad is None:
      #print("again waste")
      self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
    involved_idx = self.active_index + self.inactive_index
    probs_slice = F.softmax(torch.stack([self.AP_path_alpha[idx] for idx in involved_idx]), dim=0).data
    for i in range(2):
      for j in range(2):
        origin_i = involved_idx[i]
        origin_j = involved_idx[j]
        #print(self.AP_path_alpha.grad.data)
        #print(self.AP_path_alpha.grad)
        #print(i,j,origin_i,origin_j)
        #print(probs_slice)
        self.AP_path_alpha.grad.data[origin_i] += (
            binary_grads[origin_j] * probs_slice[j] * (delta_ij(i, j) - probs_slice[i]))
    print("GRAD OF PATHS = ", self.AP_path_alpha.grad.data[involved_idx])
    for _i, idx in enumerate(self.active_index):
      self.active_index[_i] = (idx, self.AP_path_alpha[idx])
    for _i, idx in enumerate(self.inactive_index):
      self.inactive_index[_i] = (idx, self.AP_path_alpha[idx])

  def arch_optimizer(self):
    self.arch_parameters = []
    involved_idx = self.active_index + self.inactive_index
    print("Tensor =", self.AP_path_alpha.data)
    print("INVOLVED INDEX=",involved_idx)
    self.arch_parameters = [self.AP_path_alpha.data[idx] for idx in involved_idx]
    #self.arch_parameters.append(self.AP_path_alpha[involved_idx[1]])
    #print("Tensor =", self.AP_path_alpha.data)
    print("ARCH_PARAMS BEFORE STEP=",self.arch_parameters)
    arch_optimizer = optim.Adam(self.arch_parameters, lr = 6e-3)
    return arch_optimizer

  def rescale_updated_arch_param(self):
    involved_idx = [idx for idx, _ in (self.active_index + self.inactive_index)]
    old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
    new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]
    offset = torch.log(
            sum([torch.exp(alpha) for alpha in new_alphas]) / sum([torch.exp(alpha) for alpha in old_alphas])
        )

    #self.AP_path_alpha.data[involved_idx[0]] += offset
    #self.AP_path_alpha.data[involved_idx[1]] -= offset
    for idx in involved_idx:
      self.AP_path_alpha.data[idx] -= offset
    #print("ALPHAS OF ALL PATHS =", self.AP_path_alpha.data)

  def store(grad,parent):
    print(grad,parent)
    self.grads[parent] = grad.clone()



def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0

def get_main_model(main_model_description, in_channels, mode="search_arch"):
  
  num_layers = main_model_description["num_layers"]
  dropout = main_model_description["dropout"]
  bn = main_model_description["batch_normalization"]
  activation = main_model_description["activation"]
  algorithm = main_model_description["algorithm"]["type"]
  reduction_block = main_model_description["reduction_block"]
  out_channels = in_channels
  
  
  layers = []

  if algorithm == "nad" and mode == "search_arch":
    epochs = main_model_description["algorithm"]["epochs"]
    final_prob = main_model_description["algorithm"]["final_prob"] 
    beta = torch.exp(torch.log(torch.tensor(final_prob)) / epochs)

  if mode == "final_arch":
    topology = main_model_description["topology"]

  for layer in range(1, num_layers+1):
    block_list = []

    arg = {}
    arg["in_channels"] = in_channels
    arg["out_channels"] = in_channels# // 2 depends on the type of channel concat we are using
    arg["learnable"] = mode == "search_arch"
    arg["activation"] = activation
    arg["dropout"] = dropout
    arg["bn"] = bn
    arg["name"] = "core_layer_{}_".format(layer)


    if mode == "final_arch" and str(layer) in topology.keys():
      block = main_model_description["blocks"][int(topology[str(layer)])]
      print("Adding layer {} with filter {} and channels = {}".format(layer, block["kernel_size"], in_channels))
      arg["kernel_size"] = block["kernel_size"]
      if block["block"] == "conv2d":
        block_list.append(ConvLayer(arg))
      elif block["block"] == "conv2d_depth":
        block_list.append(DepthConvLayer(arg))
      elif block["block"] == "maxpool":
        arg["type"] = "max"
        block_list.append(PoolingLayer(arg))
      elif block["block"] == "avgpool":
        arg["type"] = "avg"
        block_list.append(PoolingLayer(arg))
      else:
        print("block {} not found".format(block["block"]))
      #if (layer) % 3 == 0:
      #  previous_block = layers[layer - 2]
      #  current_block = SingleLayer(block_list, algorithm)
      #  layers.append(previous_block + current_block)
      #else:
      layers.append(SingleLayer(block_list, algorithm))

    elif mode == "search_arch":
      if algorithm=="random":
        index = np.random.choice(range(len(main_model_description["blocks"])),
          replace=False)
        block = main_model_description["blocks"][index]
        arg["kernel_size"] = block["kernel_size"]
        print("INDEX=", index, "KERNEL", block["kernel_size"])
        if block["block"] == "conv2d":
          block_list.append(ConvLayer(arg))
        elif block["block"] == "conv2d_depth":
          block_list.append(DepthConvLayer(arg))
        layers.append(SingleLayer(block_list, algorithm))
      for block in main_model_description["blocks"]:

        arg["kernel_size"] = block["kernel_size"]

        if block["block"] == "conv2d":
          block_list.append(ConvLayer(arg))
        elif block["block"] == "conv2d_depth":
          block_list.append(DepthConvLayer(arg))
        elif block["block"] == "maxpool":
          arg["type"] = "max"
          block_list.append(PoolingLayer(arg))
        elif block["block"] == "avgpool":
          arg["type"] = "avg"
          block_list.append(PoolingLayer(arg))

      if algorithm == "nad":
        layers.append(SingleLayer(block_list, algorithm, beta))
      elif algorithm == "plnas":
        layers.append(SingleLayer(block_list, algorithm))

    if layer in main_model_description["reduction_layers"]:
      block_list = []
      for block in reduction_block:
        arg["out_channels"] = in_channels
        arg["learnable"] = False
        arg["kernel_size"] = block["kernel_size"]
        arg["stride"] = block["stride"]
        arg["name"] = "reduction_layer_"

        if block["block"] == "conv2d":
          block_list.append(ConvLayer(arg))
        elif block["block"] == "maxpool":
          arg["type"] = "max"
          block_list.append(PoolingLayer(arg))

      layers.append(SingleLayer(block_list, reduction=True))
      in_channels *= 2
      print("Adding reduction layer at layer {} and channels = {}".format(layer, in_channels))

  return layers, in_channels
