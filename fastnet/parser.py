from . import net
from fastnet.layer import ConvLayer, MaxPoolLayer, AvgPoolLayer, \
  CrossMapResponseNormLayer, SoftmaxLayer, NeuronLayer, ResponseNormLayer, \
  FCLayer, DataLayer, MultiLabelOutput

from . import util
from fastnet.util import isfloat


def parse_config_file(parsing_file):
  rst = []
  with open(parsing_file) as f:
    for line in f:
      line = line.strip()
      if line.startswith('#'):
        continue
      elif line.startswith('['):
        name = line[1:line.find(']')]
        rst.append({'name': name})
      elif len(line) == 0:
        continue
      else:
        key = line[0:line.find('=')]
        value = line[line.find('=') + 1: len(line)]

        if value.isdigit():
          value = int(value)
        elif isfloat(value):
          value = float(value)

        rst[-1][key] = value
  return rst


def load_model(net, model):
  if is_cudaconvnet_config(model):
    # AlexK config file
    add_layers(CudaconvNetBuilder(), net, model)
  else:
    # FastNet config file
    add_layers(FastNetBuilder(), net, model)

  return net


def load_from_checkpoint(config, checkpoint, image_shape):
  network = net.FastNet(image_shape)
  if checkpoint is not None:
    load_model(network, checkpoint)
  else:
    load_model(network, parse_config_file(config))
  return network


# @util.lazyinit(fastnet.init)
def add_layers(builder, net, model):
  net.append_layer(DataLayer(name='data0', image_shape=net.image_shape))
  for layer in model:
    l = builder.make_layer(net, layer)
    if l is not None:
      net.append_layer(l)


class Builder(object):
  def make_layer(self, net, ld):
    if ld['type'] == 'conv':
      return self.conv_layer(ld)
    elif ld['type'] == 'pool':
      return self.pool_layer(ld)
    elif ld['type'] == 'neuron':
      return self.neuron_layer(ld)
    elif ld['type'] == 'fc':
      return self.fc_layer(ld)
    elif ld['type'] == 'softmax':
      return self.softmax_layer(ld)
    elif ld['type'] == 'multilabel':
      return self.multilabel_layer(ld)
    elif ld['type'] == 'rnorm':
      return self.rnorm_layer(ld)
    elif ld['type'] == 'crmnorm':
      return self.crm_layer(ld)
    elif ld['type'] == 'local':
      return self.local_layer(ld)
    else:
      util.log_warn('Unknown layer %s' % ld['type'])


class FastNetBuilder(Builder):
  def conv_layer(self, ld):
    num_filter = ld['numFilter']
    filterSize = ld['filterSize']
    padding = ld['padding']
    stride = ld['stride']
    initW = ld.get('initW', 0.01)
    initB = ld.get('initB', 0.00)
    epsW = ld.get('epsW', 0.001)
    epsB = ld.get('epsB', 0.002)
    if epsB == 0:
      epsB = 0.002
    momW = ld.get('momW', 0.0)
    momB = ld.get('momB', 0.0)
    # sharedBiases = ld.get('sharedBiases', default = 1)
    # partialSum = ld.get('partialSum', default = 0)
    wc = ld.get('wc', 0.0)
    name = ld['name']
    cv = ConvLayer(name=name,
                   num_filter=num_filter,
                   filterSize=filterSize,
                   padding=padding,
                   stride=stride,
                   init_w=initW,
                   init_b=initB,
                   eps_w=epsW,
                   eps_b=epsB,
                   mom_w=momW,
                   mom_b=momB,
                   wc=wc)
    return cv

  def local_layer(self, ld):
    numFilter = ld['numFilter']
    filterSize = ld['filterSize']
    padding = ld['padding']
    stride = ld['stride']
    initW = ld.get('initW', 0.01)
    initB = ld.get('initB', 0.00)
    epsW = ld.get('epsW', 0.001)
    epsB = ld.get('epsB', 0.002)
    momW = ld.get('momW', 0.0)
    momB = ld.get('momB', 0.0)
    # sharedBiases = ld.get('sharedBiases', default = 1)
    # partialSum = ld.get('partialSum', default = 0)
    wc = ld.get('wc', 0.0)
    name = ld['name']
    cv = LocalUnsharedLayer(name, numFilter, (filterSize, filterSize), padding,
                            stride, initW, initB, epsW, epsB, momW, momB, wc,
                            bias, weight,
                            weightIncr=weightIncr, biasIncr=biasIncr)
    return cv

  def pool_layer(self, ld):
    stride = ld['stride']
    start = ld['start']
    poolSize = ld['poolSize']
    name = ld['name']
    pool = ld['pool']
    if pool == 'max':
      return MaxPoolLayer(name=name, poolSize=poolSize, stride=stride,
                          start=start)
    elif pool == 'avg':
      return AvgPoolLayer(name=name, poolSize=poolSize, stride=stride,
                          start=start)
    assert False, 'Bad layer type: %s' % pool


  def crm_layer(self, ld):
    name = ld['name']
    pow = ld['pow']
    size = ld['size']
    scale = ld['scale']
    blocked = bool(ld.get('blocked', 0))
    return CrossMapResponseNormLayer(name=name,
                                     pow=pow,
                                     size=size,
                                     scale=scale,
                                     blocked=blocked)

  def softmax_layer(self, ld):
    name = ld['name']
    return SoftmaxLayer(name=name)

  def neuron_layer(self, ld):
    name = ld['name']
    if ld['neuron'] == 'relu':
      e = ld['e']
      return NeuronLayer(name=name, type='relu', e=e)

    if ld['neuron'] == 'tanh':
      a = ld['a']
      b = ld['b']
      return NeuronLayer(name, type='tanh', a=a, b=b)

    assert False, 'No implementation for the neuron type' + ld['neuron']['type']

  def rnorm_layer(self, ld):
    name = ld['name']
    pow = ld['pow']
    size = ld['size']
    scale = ld['scale']
    return ResponseNormLayer(name=name, pow=pow, size=size, scale=scale)

  def multilabel_layer(self, ld):
    return MultiLabelOutput(name='multi-out',
                            a=ld.get('a', 1.0),
                            b=ld.get('b', 1.0))

  def fc_layer(self, ld):
    epsB = ld.get('epsB', 0.002)
    if epsB == 0:
      epsB = 0.002
    epsW = ld.get('epsW', 0.001)
    initB = ld.get('initB', 0.00)
    initW = ld.get('initW', 0.01)
    momB = ld.get('momB', 0.0)
    momW = ld.get('momW', 0.0)
    wc = ld.get('wc', 0.0)
    dropRate = ld.get('dropRate', 0.0)

    n_out = ld['outputSize']
    name = ld['name']
    return FCLayer(name=name,
                   output_size=n_out,
                   eps_w=epsW,
                   eps_b=epsB,
                   init_w=initW,
                   init_b=initB,
                   mom_w=momW,
                   mom_b=momB,
                   wc=wc,
                   drop_rate=dropRate)


class CudaconvNetBuilder(FastNetBuilder):
  def conv_layer(self, ld):
    numFilter = ld['filters']
    filterSize = ld['filterSize']
    padding = ld['padding']
    stride = ld['stride']
    initW = ld['initW']
    initB = ld.get('initB', 0.0)
    name = ld['name']
    epsW = ld['epsW']
    epsB = ld['epsB']

    momW = ld['momW']
    momB = ld['momB']

    wc = ld['wc']

    bias = ld.get('biases', None)
    weight = ld.get('weights', None)
    cv = ConvLayer(name=name,
                   num_filter=numFilter,
                   filter_size=filterSize,
                   padding=padding,
                   stride=stride,
                   init_w=initW,
                   init_b=initB,
                   eps_w=epsW,
                   eps_b=epsB,
                   mom_w=momW,
                   mom_b=momB,
                   wc=wc)
    return cv

  def pool_layer(self, ld):
    stride = ld['stride']
    start = ld['start']
    poolSize = ld['sizeX']
    name = ld['name']
    pool = ld['pool']
    if pool == 'max':
      return MaxPoolLayer(name=name, pool_size=poolSize, stride=stride, start=start)
    else:
      return AvgPoolLayer(name=name, pool_size=poolSize, stride=stride, start=start)


  def neuron_layer(self, ld):
    if ld['neuron']['type'] == 'relu':
      name = ld['name']
      #e = ld['neuron']['e']
      return NeuronLayer(name=name, type='relu')
    if ld['neuron']['type'] == 'tanh':
      name = ld['name']
      a = ld['neuron']['a']
      b = ld['neuron']['b']
      return NeuronLayer(name=name, type='tanh', a=a, b=b)

    assert False, 'No implementation for the neuron type' + ld['neuron']['type']


  def fc_layer(self, ld):
    epsB = ld['epsB']
    epsW = ld['epsW']
    initB = ld.get('initB', 0.0)
    initW = ld['initW']
    momB = ld['momB']
    momW = ld['momW']

    wc = ld['wc']
    dropRate = ld.get('dropRate', 0.0)

    n_out = ld['outputs']
    bias = ld.get('biases', None)
    weight = ld.get('weights', None)

    if bias is not None:
      bias = bias.transpose()
      bias = np.require(bias, dtype = np.float32, requirements = 'C')
    if weight is not None:
      weight = weight.transpose()
      weight = np.require(weight, dtype = np.float32, requirements = 'C')

    name = ld['name']
    return FCLayer(name=name,
                   output_size=n_out,
                   eps_w=epsW,
                   eps_b=epsB,
                   init_w=initW,
                   init_b=initB,
                   mom_w=momW,
                   mom_b=momB,
                   wc=wc,
                   drop_rate=dropRate)

  def rnorm_layer(self, ld):
    name = ld['name']
    pow = ld['pow']
    size = ld['size']
    scale = ld['scale']
    scale = scale * size ** 2
    return ResponseNormLayer(name=name, pow=pow, size=size, scale=scale)

  def crm_layer(self, ld):
    name = ld['name']
    pow = ld['pow']
    size = ld['size']
    scale = ld['scale']
    scale = scale * size
    blocked = bool(ld.get('blocked', 0))
    return CrossMapResponseNormLayer(name=name, pow=pow, size=size, scale=scale, blocked=blocked)
  
def is_cudaconvnet_config(model):
 for layer in model:
   if 'filters' in layer or 'channels' in layer:
     return True
 return False

