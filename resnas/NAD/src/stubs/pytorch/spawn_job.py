import logging, os, time, sys
import torch
import numpy as np
import torch.nn as nn
import torch

class ValidBatch:
  def __init__(self, validation_queue):
    self.valid_loader = validation_queue
    self._valid_iter = None

  def valid_next_batch(self):
    if self._valid_iter is None:
      self._valid_iter = iter(self.valid_loader)
    try:
      data = next(self._valid_iter)
    except StopIteration:
      self._valid_iter = iter(self.valid_loader)
      data = next(self._valid_iter)
    return data

def save_model(path, model):
  filename = os.path.join(path, 'checkpoint.pth.tar')
  torch.save(model.state_dict(), filename)

def train(train_queue, validation_queue, weight_optimizer, arch_optimizer,
                 model, criterion, writer, use_gpu=True, epoch=None, grad_clip_fn=None, aux_head=None):
  model.train()

  log_image_step = 300
  total_wloss = torch.tensor(0.)
  total_aloss = torch.tensor(0.)
  avg_wloss=0.0
  avg_aloss=0.0
  if use_gpu:
    total_wloss = total_wloss.cuda()
    total_aloss = total_aloss.cuda()
  validation_batch = ValidBatch(validation_queue)

  for iteration, (inputs, targets) in enumerate(train_queue):
    model.set_phase("train")
    if use_gpu:
      inputs = inputs.cuda()
      targets = targets.cuda()
    logits = model(inputs)
    loss = criterion(logits, targets)
    #print(inputs.size())
    total_wloss += loss.mean()
    model.zero_grad()
    loss.backward()
    grad_clip_fn(model.parameters())
    weight_optimizer.step()
    if iteration % log_image_step == 0:
      writer.add_images("train_images", inputs, (epoch * len(train_queue)) + iteration)
    avg_wloss = (total_wloss / (iteration + 1)).item()
    if epoch>=0 and iteration%2==0 :
      model.set_phase("validation")
      #print("VALIDATION SIZE", validation_batch.valid_next_batch()[0].size())
      images, labels = validation_batch.valid_next_batch()
      if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
      for layer in model.layers:
        if type(layer).__name__ == 'SingleLayer' and layer.reduction==False:
          layer.reset_binary_gates()
          layer.unused_modules_off()

      logits = model(images)
      loss = criterion(logits, labels)
      total_aloss += loss.mean()
      model.zero_grad()
      loss.backward()
      for layer in model.layers:
        if type(layer).__name__ == 'SingleLayer' and layer.reduction==False:
          layer.set_arch_param_grad()
      arch_optimizer.step()
      for layer in model.layers:
        if type(layer).__name__ == 'SingleLayer' and layer.reduction==False:
          layer.rescale_updated_arch_param()
          layer.unused_modules_back()
      #del loss, logits, images, labels
      avg_aloss = (total_aloss / (iteration + 1)).item()
  #del loss, logits
  return avg_wloss , avg_aloss

def weight_train(train_queue, model, optimizer, criterion, writer,
                 use_gpu=False, epoch=None, grad_clip_fn=None, aux_head=None):

  model.module.set_phase("train")
  #model.set_phase("train")
  model.train()

  log_image_step = 300
  total_loss = torch.tensor(0.)
  optimizer.zero_grad()

  if use_gpu:
    total_loss = total_loss.cuda()

  for iteration, (inputs, targets) in enumerate(train_queue):
    #print("INPUT SIZE BEFORE FORWARD", inputs.size())
    if use_gpu:
      inputs = inputs.cuda()
      targets = targets.cuda()
    logits = model(inputs)

    if aux_head:
      logits, aux_logits = logits
      aux_loss = 0.4 * criterion(aux_logits, targets)

    loss = criterion(logits, targets)

    total_loss += loss.mean()

    if aux_head:
      loss += aux_loss
      total_loss += aux_loss.mean()

    loss.backward()
    grad_clip_fn(model.parameters())
    optimizer.step()
    optimizer.zero_grad()
    del loss, logits
    if iteration % log_image_step == 0:
      writer.add_images("train_images", inputs, (epoch * len(train_queue)) + iteration)

  avg_loss = (total_loss / (iteration + 1))

  return avg_loss

def arch_train(validation_queue, model, optimizer, criterion, use_gpu=False, epoch=None, aux_head=None):

  #model.module.set_phase("validation")
  model.set_phase("validation")
  total_loss = torch.tensor(0.).cuda()
  optimizer.zero_grad()
  for iteration, (inputs, targets) in enumerate(validation_queue):
    
    if use_gpu:
      inputs = inputs.cuda()
      targets = targets.cuda()

    for layer in model.layers:
      if type(layer).__name__ == 'SingleLayer' and layer.reduction==False:
        layer.reset_binary_gates()
        layer.unused_modules_off()
        #a_optimizer = layer.arch_optimizer()
        #a_optimizer.zero_grad()
        #arc_optimizer.append(a_optimizer)
    logits = model(inputs)

    if aux_head:
      logits, aux_logits = logits
      aux_loss = 0.4 * criterion(aux_logits, targets)

    loss = criterion(logits, targets)

    total_loss += loss.mean()

    if aux_head:
      loss += aux_loss
      total_loss += aux_loss.mean()
    #idx = 0
    loss.backward()
    for layer in model.layers:
      if type(layer).__name__ == 'SingleLayer' and layer.reduction==False:
        layer.set_arch_param_grad()
        #arc_optimizer[idx].step()
    optimizer.step()
    for layer in model.layers:
      if type(layer).__name__ == 'SingleLayer' and layer.reduction==False:
        #print("ARCH_PARAMS_AFTER STEP = ",layer.arch_parameters) 
        layer.rescale_updated_arch_param()
        layer.unused_modules_back()
    optimizer.zero_grad()
    del loss, logits

  avg_loss = (total_loss / (iteration + 1))

  return avg_loss

def test(test_queue, model, use_gpu=False, epoch=None):

  model.eval()
  model.module.set_phase("test")

  total_acc = 0
  total_elements = 0
  for iteration, (test_inputs, test_labels) in enumerate(test_queue):
    if use_gpu:
      test_inputs = test_inputs.cuda()
      test_labels = test_labels.cuda()
    with torch.no_grad():
      test_logits = model(test_inputs)
      torch_sm = nn.Softmax()
      test_logits_sm = torch_sm(test_logits)
      _, test_pred = torch.max(test_logits_sm, 1)

    compare = torch.eq(test_pred, test_labels)
    total_acc += (torch.sum(compare).float())
    total_elements += np.shape(compare)[0]

  return (total_acc / total_elements).item()

def spawn_job(components):

  parameters = components["parameters"]
  w_scheduler = components["w_scheduler"]
  #arch_scheduler = components["arch_scheduler"]
  mode = components["mode"]
  writer = components["writer"]
  grad_clip_fn = lambda params: nn.utils.clip_grad_norm_(params,
                                   components["gradient_clipping"]["value"], norm_type=2)
  aux_head = components.get("aux_head")

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p %Z')
  logging.getLogger().setLevel(logging.INFO)

  output_path = parameters["output_dir"] + parameters["trial_name"] + "/"
  if not os.path.exists(output_path):
    os.mkdir(output_path)
  
  log_file = output_path + 'exp.log'
  if not os.path.exists(log_file):
    open(log_file, 'a').close()

  fh = logging.FileHandler(os.path.join(log_file))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)


  for epoch in range(parameters["epochs"]):

    #scheduler.step()
    #scheduler.get_lr()
    components["model"].module.set_epoch(epoch)

    weight_lr = components["weight_optimizer"].param_groups[0]["lr"]
    if mode == "search_arch":
      arch_lr = 0.006 #components["arch_optimizer"].param_groups[0]["lr"]
      logging.info("Epoch number {}, weight_lr = {:.4f} and arch_lr = {:.4f}".format(epoch, weight_lr, arch_lr))
      writer.add_scalar('data/weight_lr', weight_lr, epoch)
      writer.add_scalar('data/arch_lr', arch_lr, epoch)
    else:
      logging.info("Epoch number {} and weight_lr = {:.4f}".format(epoch, weight_lr))
      writer.add_scalar('data/weight_lr', weight_lr, epoch)

    #weight_loss = weight_train(components["train_queue"], components["model"],
     # components["weight_optimizer"], components["criterion"], components["writer"],
      #parameters["use_gpu"], torch.tensor(epoch), grad_clip_fn, aux_head)

    #if mode == "search_arch":
     # arch_loss = arch_train(components["valid_queue"], components["model"],
      #  components["arch_optimizer"], components["criterion"],
       # parameters["use_gpu"], torch.tensor(epoch), aux_head)

    if mode == "search_arch":
      weight_loss, arch_loss = train(components["train_queue"], components["valid_queue"], components["weight_optimizer"],
        components["arch_optimizer"], components["model"], components["criterion"], components["writer"],
        parameters["use_gpu"], torch.tensor(epoch), grad_clip_fn, aux_head)

    else:
      weight_loss = weight_train(components["train_queue"], components["model"],
        components["weight_optimizer"], components["criterion"], components["writer"],
        parameters["use_gpu"], torch.tensor(epoch), grad_clip_fn, aux_head)

    accuracy = test(components['test_queue'], components["model"],
         parameters["use_gpu"], torch.tensor(epoch))
    #if epoch<300:
    w_scheduler.step()
    #else:
      #torch.optim.lr_scheduler.ReduceLROnPlateau(components["weight_optimizer"], mode='min', factor=0.9, verbose=True).step(weight_loss)
    #arch_scheduler.step()
    if mode == "search_arch":
      #logging.info("network loss={:.3f}, accuracy={:.4f}".format(weight_loss, accuracy))
      logging.info("network loss={:.3f}, arch loss={:.4f}, accuracy={:.4f}".format(weight_loss, arch_loss, accuracy))
      writer.add_scalar('data/arch_loss', arch_loss, epoch)
    else:
      logging.info("network loss={:.2f}, accuracy={:.4f}".format(weight_loss, accuracy))

    writer.add_scalar('data/weight_loss', weight_loss, epoch)
    writer.add_scalar('data/accuracy', accuracy, epoch)

    if epoch % parameters["save_frequency"] == 0:
      save_model(output_path, components["model"])
