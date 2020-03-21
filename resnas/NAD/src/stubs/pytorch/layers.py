import torch
import torch.nn as nn

class BasicUnit(nn.Module):

  def __init__(self):
    super(BasicUnit, self).__init__()
    self.phase = "train"
    self.epoch = 0

  def set_phase(self, phase):
    self.phase = phase

  def set_epoch(self, epoch):
    self.epoch = epoch

  def forward(self, x):
    raise NotImplementedError

class Layer(BasicUnit):
  """Basic Layer
  default input expectation is a 4D Tensor feature map
  that's why Dropout2d and BatchNorm2d is used"""
  def __init__(self, arg):
    super(Layer, self).__init__()

    self.__name__ = arg['name']
 
    if 'activation' in arg.keys():
      if arg['activation'] == 'relu':
        self.activation = nn.ReLU()
    else:
      self.activation = False

    if 'dropout' in arg.keys():
      self.dropout = nn.Dropout2d(p=arg['dropout'])
    else:
      self.dropout = False

    if 'bn' in arg.keys():
      self.bn = nn.BatchNorm2d(arg['out_channels'])
    else:
      self.bn = False


  def weight_op(self, x):
    raise NotImplementedError

  def forward(self, x):
    """
    For now use a fix order of weight_op -> BN -> activation
    """
    x = self.weight_op(x)

    if self.bn:
      x = self.bn(x)

    if self.dropout and self.phase == "train":
      x = self.dropout(x)

    if self.activation:
      x = self.activation(x)
    

    return x

class ConvLayer(Layer):
  """
  L is a special layer type with following order
    Relu -> Conv -> Batch Normalization
  """
  def __init__(self, arg):
    self.stride = arg.get('stride') or 1
    self.kernel_size = arg['kernel_size']
    self.out_channels = arg['out_channels']

    arg['name'] += 'conv_{}x{}'.format(self.kernel_size[0], self.kernel_size[1])

    super(ConvLayer, self).__init__(arg)

    self.conv = nn.Conv2d(arg['in_channels'], arg['out_channels'],
      kernel_size=self.kernel_size, stride=self.stride, bias=False)

  def weight_op(self, x):
    x = self.conv(x)
    return x

class DepthConvLayer(Layer):
  """
  L is a special layer type with following order
    Relu -> Conv -> Batch Normalization
  """
  def __init__(self, arg):
    self.stride = arg.get('stride') or 1
    self.kernel_size = arg['kernel_size']
    self.out_channels = arg['out_channels']

    arg['name'] += 'sep_conv_{}x{}'.format(self.kernel_size[0], self.kernel_size[1])

    super(DepthConvLayer, self).__init__(arg)

    self.depthwise = nn.Conv2d(arg['in_channels'], arg['in_channels'],
      kernel_size=self.kernel_size, stride=self.stride, groups=arg['in_channels'], bias=False)
    self.pointwise = nn.Conv2d(arg['in_channels'], arg['out_channels'], kernel_size=1, bias=False)

  def weight_op(self, x):
    x = self.depthwise(x)
    x = self.pointwise(x)
    return x

class PoolingLayer(Layer):
  """docstring for PoolingLayer"""
  def __init__(self, arg):
    self.stride = arg.get('stride') or 1
    self.kernel_size = arg['kernel_size']

    self.out_channels = arg['out_channels']
    arg['name'] += '{}pool_{}x{}'.format(arg["type"], self.kernel_size[0], self.kernel_size[1])

    super(PoolingLayer, self).__init__(arg)
    
    if arg["type"] == "avg":
      self.pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)
    elif arg["type"] == "max":
      self.pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)

  def weight_op(self, x):
    x = self.pool(x)
    return x
    
class FullyConnectedLayer(Layer):
  """
  """
  def __init__(self, arg):

    arg['name'] += 'fc_{}x{}'.format(arg['num_features'], arg['out_features'])
    super(FullyConnectedLayer, self).__init__(arg)

    if "dropout" in arg.keys():
      self.dropout = nn.Dropout(arg['num_features'])
    if "bn" in arg.keys():
      self.bn = nn.BatchNorm1d(arg['num_features'])

    self.layer = nn.Linear(arg['num_features'], arg['out_features'])

  def weight_op(self, x):
    x = self.layer(x)
    return x

class GlobalAvgPooling(BasicUnit):
  """
  """
  def __init__(self, arg):
    arg['name'] += 'global_avg_pooling'
    super(GlobalAvgPooling, self).__init__()

  def forward(self, x):
    x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
    return x
