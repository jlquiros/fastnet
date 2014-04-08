import numpy as np
import time

from pycuda import gpuarray, driver

from . import util
from fastnet.layer import TRAIN, WeightedLayer, TEST


def to_gpu(array):
  assert array.dtype == np.float32
  if isinstance(array, gpuarray.GPUArray): return array
  return gpuarray.to_gpu(array).astype(np.float32)

class FastNet(object):
  def __init__(self, image_shape):
    self.layers = []
    self.image_shape = image_shape
    self.counts = None

  def __getitem__(self, name):
    for layer in self.layers:
      if layer.name == name:
        return layer

  def __iter__(self):
    return iter(self.layers)

  def append_layer(self, layer):
    if self.layers:
      layer.attach(self.layers[-1])

    self.layers.append(layer)
    util.log_info('Append: %s  [%s] : %s',
                  layer.name,
                  layer.type,
                  layer.get_output_shape())
    return layer

  def fprop(self, data, train=TRAIN):
    assert data.dtype == np.float32
    data = to_gpu(data)
    assert len(self.layers) > 0, 'No outputs: uninitialized network!'

    input = data
    for layer in self.layers:
      # util.log_info('Fprop: %s', layer.name)
      st = time.time()
      output = layer.fprop(input, train)
      driver.Context.synchronize()
      ed = time.time()
      # print 'fprop', layer.name, ed - st

      input = layer.output
    
    return self.layers[-1].output

  def bprop(self, label, train=TRAIN):
    grad = label
    for i in range(1, len(self.layers)):
      curr = self.layers[-i]
      if not curr.should_bprop: return
      prev = self.layers[-(i + 1)]     
      st = time.time()
      curr.bprop(grad, prev.output, curr.output, prev.output_grad)
      driver.Context.synchronize()
      ed = time.time()
      # print 'bprop', curr.name, ed - st
      grad = prev.output_grad

  def update(self):
    for layer in self.layers:
      layer.update()
  
  def check_weights(self):
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        assert not np.any(np.isnan(layer.weight.wt.get()))
        assert not np.any(np.isnan(layer.bias.wt.get()))

  def adjust_learning_rate(self, factor=1.0):
    util.log_info('Adjusting learning rate: %s', factor)
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        layer.weight.epsilon *= factor
        layer.bias.epsilon *= factor

    self.print_learning_rates()

  def set_learning_rate(self, eps_w, eps_b):
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        layer.weight.epsilon = eps_w
        layer.bias.epsilon = eps_b
    self.print_learning_rates()

  def print_learning_rates(self):
    util.log('Learning rates:')
    for layer in self.layers:
      if isinstance(layer, WeightedLayer):
        util.log('%s: %s %s %s', layer.name, layer.__class__.__name__,
                 layer.weight.epsilon, layer.bias.epsilon)
      else:
        util.log('%s: %s', layer.name, layer.__class__.__name__)

  def clear_weight_incr(self):
    for l in self.layers:
      if isinstance(l, WeightedLayer):
        l.clear_incr()

  def get_cost(self, label, prediction):
    cost_layer = self.layers[-1]
    assert not np.any(np.isnan(prediction.get()))
    cost = cost_layer.logreg_cost(label, prediction)
    batch_correct = cost_layer.get_correct()
    return cost, batch_correct

  def get_cost_multiview(self, label, prediction, num_view):
    cost_layer = self.layers[-1]
    assert not np.any(np.isnan(prediction.get()))
    cost_layer.logreg_cost_multiview(label, prediction, num_view)
    return cost_layer.cost.get().sum(), cost_layer.batchCorrect

  def get_batch_information(self):
    return self.cost, self.correct
  
  def get_correct(self):
    return self.layers[-1].get_correct()

  def train_batch(self, data, label, train=TRAIN,):
    data = to_gpu(data)
    label = to_gpu(label)

    prediction = self.fprop(data, train)
    self.cost, self.correct = self.get_cost(label, prediction)

    if train == TRAIN:
      self.bprop(label)
      self.update()

    # make sure we have everything finished before returning!
    # also, synchronize properly releases the Python GIL,
    # allowing other threads to make progress.
    driver.Context.synchronize()

  def test_batch_multiview(self, data, label, num_view):
    data = to_gpu(data)
    label = to_gpu(label)
    label = label.reshape((label.size, 1))
    
    prediction = self.fprop(data, TEST)
    cost, correct = self.get_cost_multiview(label, prediction, num_view)
    self.cost += cost
    self.correct += correct
    driver.Context.synchronize()

  def get_dumped_layers(self):
    return [l.dump() for l in self.layers]

  def get_report(self):
    pass

  def get_image_shape(self):
    return self.layers[0].get_output_shape()

  def get_learning_rate(self):
    return self.learning_rate

  def get_layer_by_name(self, layer_name):
    for l in self.layers:
      if l.name == layer_name:
        return l

    raise KeyError, 'Missing layer: %s' % layer_name

  def get_output_by_name(self, layer_name):
    return self.get_layer_by_name(layer_name).output

  def get_output_index_by_name(self, layer_name):
    for idx, l in enumerate(self.layers):
      if l.name == layer_name:
        return idx

    raise KeyError, 'Missing layer: %s' % layer_name

  def get_output_by_index(self, index):
    return self.layers[index].output

  def get_first_active_layer_name(self):
    for layer in self.layers:
      if layer.disable_bprop == False and isinstance(layer, WeightedLayer):
        return layer.name
    return ''

  def get_weight_by_name(self, name):
    l = self.get_layer_by_name(name)
    return l.weight.wt.get() + l.bias.wt.get().transpose()

  def get_summary(self):
    sum = []
    for l in self.layers:
      if isinstance(l, WeightedLayer):
        sum.append(l.get_summary())
    return sum
