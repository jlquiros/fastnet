import numpy as np

from parakeet import jit
from pycuda import gpuarray
from traits.api import (
  HasStrictTraits, Any, Float, Int, Bool, Tuple, String, Instance, This)

import cudaconv2
from fastnet.cuda_kernel import (
  gpu_copy_to, add_vec_to_rows, add_row_sum_to_vec, dot, bigger_than_scaler,
  transpose, col_max_reduce, add_vec_to_cols, eltwise_exp, add_col_sum_to_vec,
  div_vec_to_cols, find_col_max_id, same_reduce, logreg_cost_col_reduce,
  softmax_bprop, relu_activate, relu_compute_grad, tanh_activate,
  tanh_compute_grad, same_reduce_multiview, gpu_partial_copy_to)
from fastnet.util import divup
from fastnet.weights import WEIGHTS, Weight, to_gpu
from . import util


PFout = False
PBout = False
TEST = 0
TRAIN = 1

# seed = 0
# np.random.seed(seed)
rng = np.random.RandomState()


def col_randn(shape, dtype):
  rand_array = rng.standard_normal(np.prod(shape))
  return np.require(
    rand_array.reshape(shape),
    dtype=np.float32,
    requirements='C')


class Layer(HasStrictTraits):
  name = String()
  type = String()
  should_bprop = Bool(True)

  output = Any()
  output_grad = Any()

  batch_size = Int()
  num_color = Int()
  img_size = Int()
  output_size = Int()

  _prev = This

  def __init__(self, *args, **kw):
    HasStrictTraits.__init__(self, *args, **kw)
    self.output = None
    self.output_grad = None

  def attach(self, prev):
    self._prev = prev
    self._attach(prev)

  def _attach(self, prev):
    pass

  def update(self):
    pass

  def reset(self):
    pass

  def fprop(self, input, train=TRAIN):
    self.init_output(input)
    self._fprop(input, self.output, train)
    return self.output

  def init_output(self, input):
    self.batch_size = input.shape[-1]
    out_shape = self.get_output_shape()
    rows = int(np.prod(out_shape[:3]))
    cols = out_shape[3]

    if not self.output or self.output.shape != (rows, cols):
      #       util.log_info('Allocating... %s %s %s', rows, cols, rows * cols * 4 / 1e6)
      self.output = gpuarray.GPUArray((rows, cols), dtype=np.float32)
      self.output_grad = gpuarray.GPUArray((rows, cols), dtype=np.float32)

  def __getstate__(self):
    d = dict(self.__dict__)
    d['output'] = None
    d['output_grad'] = None
    return d


class DataLayer(Layer):
  type = 'data'
  image_shape = Tuple()

  def __init__(self, *args, **kw):
    Layer.__init__(self, *args, **kw)
    self.batch_size = self.image_shape[-1]

  def _attach(self, prev):
    assert False, 'Must be first layer!'

  def init_output(self, input):
    self.output = input
    self.output_grad = None

  def _fprop(self, input, output, train=TRAIN):
    # util.log_info("%s %s", input.shape, output.shape)
    return input
    # gpu_copy_to(input, output)

  def bprop(self, grad, input, output, out_grad):
    pass

  def get_output_shape(self):
    return tuple(list(self.image_shape[:3]) + [self.batch_size])


class WeightedLayer(Layer):
  init_w = Float()
  init_b = Float()
  eps_w = Float(0.001)
  eps_b = Float(0.002)
  mom_w = Float(0.9)
  mom_b = Float(0.9)
  wc = Float(0.004)

  weight = Instance(Weight)
  bias = Instance(Weight)

  def __init__(self, *args, **kw):
    Layer.__init__(self, *args, **kw)

    self.weight = WEIGHTS.empty('weight.' + self.name, self.eps_w, self.mom_w,
                                self.wc)
    self.bias = WEIGHTS.empty('bias.' + self.name, self.eps_b, self.mom_b, 0.0)

  def _init_weights(self, weight_shape, bias_shape):
    self.bias.shape = bias_shape
    self.weight.shape = weight_shape

    if self.weight.wt is None:
      self.weight.set_weight(
        to_gpu(col_randn(weight_shape, np.float32) * self.init_w))

    if self.bias.wt is None:
      self.bias.set_weight(
        to_gpu((np.ones(bias_shape, dtype=np.float32) * self.init_b)))

  def clear_weight_incr(self):
    self.weight.incr.fill(0)

  def clear_bias_incr(self):
    self.bias.incr.fill(0)

  def clear_incr(self):
    self.clear_weight_incr()
    self.clear_bias_incr()

  def reset(self):
    self.weight.reset()
    self.bias.reset()

  def update(self):
    if not self.should_bprop:
      return

    self.weight.update(self.batch_size)
    self.bias.update(self.batch_size)

  def get_summary(self, type='mean'):
    w = self.weight.wt.get()
    w = np.mean(np.abs(w))
    w_variance = np.var(np.abs(w.ravel()))

    b = self.bias.wt.get()
    b = np.mean(np.abs(b))
    b_variance = np.var(np.abs(b.ravel()))
    return self.name, (
      w, w_variance, b, b_variance, self.weight.epsilon, self.bias.epsilon)


class ConvLayer(WeightedLayer):
  name = String()
  num_filter = Int()
  filter_size = Int()
  padding = Int(2)
  stride = Int(1)
  tmp = Instance(gpuarray.GPUArray)

  modules = Int()

  def _attach(self, prev_layer):
    image_shape = prev_layer.get_output_shape()
    self.num_color, self.img_size, _, self.batch_size = image_shape
    self.output_size = 1 + divup(
      2 * self.padding + self.img_size - self.filter_size, self.stride)
    self.modules = self.output_size ** 2

    weight_shape = (
      self.filter_size * self.filter_size * self.num_color, self.num_filter)
    bias_shape = (self.num_filter, 1)

    self._init_weights(weight_shape, bias_shape)

  def get_cross_width(self):
    return self.filter_size - 1

  def get_single_img_size(self):
    return self.modules * self.num_filter

  def get_output_shape(self):
    return (
      self.num_filter, self.output_size, self.output_size, self.batch_size)

  def _fprop(self, input, output, train=TRAIN):
    cudaconv2.convFilterActs(input, self.weight.wt, output, self.img_size,
                             self.output_size,
                             self.output_size, -self.padding, self.stride,
                             self.num_color, 1)

    # util.log_info('%s', output.get().mean())
    self.tmp = gpuarray.empty((self.num_filter,
                               self.get_single_img_size() * self.batch_size / self.num_filter),
                              dtype=np.float32)

    gpu_copy_to(output, self.tmp)
    add_vec_to_rows(self.tmp, self.bias.wt)
    gpu_copy_to(self.tmp, output)

  def bprop(self, grad, input, output, out_grad):
    self.weight.grad.fill(0)
    self.bias.grad.fill(0)

    # bprop to next layer
    if out_grad is not None:
      cudaconv2.convImgActs(grad, self.weight.wt, out_grad, self.img_size,
                            self.img_size,
                            self.output_size, -self.padding, self.stride,
                            self.num_color, 1, 0.0, 1.0)

    # bprop weight
    cudaconv2.convWeightActs(input, grad, self.weight.grad, self.img_size,
                             self.output_size,
                             self.output_size, self.filter_size, -self.padding,
                             self.stride, self.num_color, 1, 0, 0, 1)

    # bprop bias
    gpu_copy_to(grad, self.tmp)
    add_row_sum_to_vec(self.bias.grad, self.tmp)


class PoolLayer(Layer):
  pool_size = Int()
  start = Int()
  stride = Int()

  def _attach(self, prev):
    image_shape = prev.get_output_shape()
    self.num_color, self.img_size, _, self.batch_size = image_shape
    self.output_size = divup(self.img_size - self.pool_size - self.start,
                             self.stride) + 1
    assert self.num_color % 16 == 0, \
      'Pool layers require colors to be a multiple of 16: got %s' % self.num_color

  def get_output_shape(self):
    return self.num_color, self.output_size, self.output_size, self.batch_size

  def get_cross_width(self):
    return self.pool_size - 1


class MaxPoolLayer(PoolLayer):
  def _fprop(self, input, output, train=TRAIN):
    cudaconv2.convLocalMaxPool(input, output, self.num_color, self.pool_size,
                               self.start, self.stride,
                               self.output_size)

  def bprop(self, grad, input, output, out_grad):
    cudaconv2.convLocalMaxUndo(input, grad, output, out_grad, self.pool_size,
                               self.start, self.stride, self.output_size, 0.0,
                               1.0)


class AvgPoolLayer(PoolLayer):
  def _fprop(self, input, output, train=TRAIN):
    cudaconv2.convLocalAvgPool(input, output, self.num_color, self.pool_size,
                               self.start, self.stride,
                               self.output_size)

  def bprop(self, grad, input, output, out_grad):
    cudaconv2.convLocalAvgUndo(grad, out_grad, self.pool_size,
                               self.start, self.stride, self.output_size,
                               self.img_size, 0.0, 1.0)


class ResponseNormLayer(Layer):
  pow = Float()
  size = Int()
  scale = Float()
  scaler = Float()
  denom = Instance(gpuarray.GPUArray)

  def _attach(self, prev):
    image_shape = prev.get_output_shape()
    self.num_color, self.img_size, _, self.batch_size = image_shape
    self.scaler = self.scale / self.size

  def get_output_shape(self):
    return (self.num_color, self.img_size, self.img_size, self.batch_size)

  def _fprop(self, input, output, train=TRAIN):
    self.denom = gpuarray.zeros_like(input)
    cudaconv2.convResponseNorm(input, self.denom, output, self.num_color,
                               self.size, self.scaler,
                               self.pow)

  def get_cross_width(self): return self.size - 1

  def bprop(self, grad, input, output, out_grad):
    cudaconv2.convResponseNormUndo(grad, self.denom, input, output, out_grad,
                                   self.num_color,
                                   self.size, self.scaler, self.pow, 0.0, 1.0)


class CrossMapResponseNormLayer(ResponseNormLayer):
  blocked = Int()

  def __init__(self, *args, **kw):
    ResponseNormLayer.__init__(self, *args, **kw)
    self.type = 'cmrnorm'
    self.scaler = self.scale / self.size
    util.log("pow:%s size:%s, scale:%s scaler:%s", self.pow, self.size,
             self.scale, self.scaler)

  def get_cross_width(self): return self.size - 1

  def _fprop(self, input, output, train=TRAIN):
    self.denom = gpuarray.zeros_like(input)
    cudaconv2.convResponseNormCrossMap(input, self.denom, output,
                                       self.num_color,
                                       self.size, self.scaler, self.pow,
                                       self.blocked)

  def bprop(self, grad, input, output, out_grad):
    cudaconv2.convResponseNormCrossMapUndo(grad, self.denom, input, output,
                                           out_grad, self.num_color,
                                           self.size, self.scaler, self.pow,
                                           self.blocked, 0.0, 1.0)


class FCLayer(WeightedLayer):
  output_size = Int()
  drop_rate = Float()
  input_size = Int()
  drop_mask = Instance(gpuarray.GPUArray, transient=True)

  def _attach(self, prev):
    input_shape = prev.get_output_shape()
    self.input_size = int(np.prod(input_shape[0:3]))
    self.batch_size = input_shape[3]
    weight_shape = (self.output_size, self.input_size)
    bias_shape = (self.output_size, 1)
    self._init_weights(weight_shape, bias_shape)


  def get_input_size(self):
    return self.input_size

  def get_output_shape(self):
    return self.output_size, 1, 1, self.batch_size

  def _fprop(self, input, output, train=TRAIN):
    gpu_copy_to(dot(self.weight.wt, input), output)
    add_vec_to_rows(output, self.bias.wt)

    if train == TEST:
      if self.drop_rate > 0.0:
        output *= (1.0 - self.drop_rate)
    else:
      if self.drop_rate > 0.0:
        self.drop_mask = to_gpu(
          np.random.uniform(0, 1, output.size).astype(np.float32).reshape(
            output.shape))
        bigger_than_scaler(self.drop_mask, self.drop_rate)
        gpu_copy_to(output * self.drop_mask, output)

  def bprop(self, grad, input, output, out_grad):
    if self.drop_rate > 0.0:
      gpu_copy_to(grad * self.drop_mask, grad)

    gpu_copy_to(transpose(dot(transpose(grad), self.weight.wt)), out_grad)

    self.weight.set_grad(dot(grad, transpose(input)))
    add_row_sum_to_vec(self.bias.grad, grad, alpha=0.0)


class SoftmaxLayer(Layer):
  batch_correct = Float(0)
  input_size = Int()
  cost = Float()

  def _attach(self, prev_layer):
    input_shape = prev_layer.get_output_shape()
    self.input_size, self.batch_size = int(np.prod(input_shape[0:3])), \
                                       input_shape[3]
    self.output_size = self.input_size
    self.cost = -1

  def get_output_shape(self):
    return self.output_size, 1, 1, self.batch_size

  def _fprop(self, input, output, train=TRAIN):
    y = input.get()
    y -= y.max(axis=0)
    yexp = np.exp(y)
    softmax = yexp / np.sum(yexp, axis=0)
    output.set(softmax)

  def bprop(self, label, input, output, outGrad):
    label = label.get().ravel().astype(np.int32)
    cpu_grad = np.zeros(outGrad.shape, dtype=np.float32)
    num_labels, num_examples = outGrad.shape
    output = output.get()
    input_bits = np.zeros(output.shape, dtype=np.float32)
    for i in range(label.size):
      input_bits[label[i], i] = 1
    outGrad.set(input_bits - output)
    #softmax_bprop(output, label, outGrad)

  def logreg_cost(self, label, output):
    output = output.get()
    maxid = output.argmax(axis=0).ravel().astype(np.int32)
    label = label.get().ravel().astype(np.int32)

    self.batch_correct = float(np.count_nonzero(label == maxid))
    cost = []
    for i in xrange(output.shape[1]):
      cost.append(np.log(output[label[i], i]))
    return -np.sum(cost) / label.size

  def logreg_cost_multiview(self, label, output, num_view):
    unit = self.batch_size / num_view
    if self.cost.shape[0] != unit:
      self.cost = gpuarray.zeros((unit, 1), dtype = np.float32)
    maxid = gpuarray.zeros((self.batch_size, 1), dtype = np.float32)
    find_col_max_id(maxid, output)
    self.batchCorrect = same_reduce_multiview(label, maxid, num_view)
    tmp = gpuarray.zeros((output.shape[0], unit), dtype = np.float32)
    gpu_partial_copy_to(output, tmp, 0, output.shape[0], 0, unit)
    logreg_cost_col_reduce(tmp, label, self.cost)

  def get_correct(self):
    return 1.0 * self.batch_correct / self.batch_size


@jit
def _multilabel_correct(labels, output):
  num_tags, num_examples = labels.shape
  count = 0
  for i in xrange(num_examples):
    for j in xrange(num_tags):
      if labels[i, j] == 1:
        if output[i, j] > 0.9:
          count += 1
  return count


class MultiLabelOutput(Layer):
  a = Float(1.0)
  b = Float(1.0)

  def _attach(self, prev):
    input_shape = prev.get_output_shape()
    self.input_size, self.batch_size = int(np.prod(input_shape[0:3])), \
                                       input_shape[3]
    self.output_size = self.input_size
    self.input_shape = input_shape
    self.cost = gpuarray.zeros((self.batch_size, 1), dtype=np.float32)
    self.batch_correct = 0
    self.num_tags = 0

  def get_output_shape(self):
    return self.output_size, 1, 1, self.batch_size

  def _fprop(self, input, output, train=TRAIN):
    max = gpuarray.zeros((1, self.batch_size), dtype=np.float32)
    col_max_reduce(max, input)
    add_vec_to_cols(input, max, output, alpha=-1)
    eltwise_exp(output)
    sum = gpuarray.zeros(max.shape, dtype=np.float32)
    add_col_sum_to_vec(sum, output, alpha=0)
    div_vec_to_cols(output, sum)

  def bprop(self, label, input, output, out_grad):
    label = label.get()
    output = output.get()
    return out_grad.set(label - output)

  def get_correct(self):
    return float(self.batch_correct) / self.num_tags

  def logreg_cost(self, labels, output):
    labels = labels.get()
    output = output.get()

    self.num_tags = np.sum(labels > 0)
    num_tags, num_examples = labels.shape
    correct = 0
    for i in range(num_examples):
      tags_for_example = np.sum(labels[:, i])
      label_idx = np.argsort(labels[:, i])[-tags_for_example]
      output_idx = np.argsort(output[:, i])[-tags_for_example]
      matching = np.intersect1d(label_idx, output_idx)
      correct += matching.size

    self.batch_correct = correct
    return np.sum(np.abs(labels - output))


class Neuron(object):
  def __init__(self, type):
    self.type = type

  def activate(self, input, output):
    assert False, 'No Implementation of Activation'

  def computeGrad(self, grad, output, inputGrad):
    assert False, 'No Implementation of Gradient'


class ReluNeuron(Neuron):
  def __init__(self, e):
    Neuron.__init__(self, 'relu')
    self.e = e;

  def activate(self, input, output):
    relu_activate(input, output, self.e)

  def computeGrad(self, grad, output, out_grad):
    relu_compute_grad(grad, output, out_grad, self.e)


class TanhNeuron(Neuron):
  def __init__(self, a, b):
    Neuron.__init__(self, 'tanh')
    self.a, self.b = a, b

  def activate(self, input, output):
    tanh_activate(input, output, self.a, self.b)

  def computeGrad(self, grad, output, out_grad):
    tanh_compute_grad(grad, output, out_grad, self.a, self.b)


class NeuronLayer(Layer):
  a = Float(1.0)
  b = Float(1.0)
  e = Float(0.0)
  type = String()
  neuron = Instance(Neuron)
  image_shape = Tuple()

  def __init__(self, *args, **kw):
    Layer.__init__(self, *args, **kw)
    if self.type == 'relu':
      self.neuron = ReluNeuron(self.e)
    elif self.type == 'tanh':
      self.neuron = TanhNeuron(self.a, self.b)

  def get_cross_width(self):
    return 0

  def get_output_shape(self):
    return self._prev.get_output_shape()

  def _fprop(self, input, output, train=TRAIN):
    self.neuron.activate(input, output)

  def bprop(self, grad, input, output, out_grad):
    self.neuron.computeGrad(grad, output, out_grad)

