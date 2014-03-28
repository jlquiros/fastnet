from PIL import Image
from os.path import basename
from fastnet import util
import Queue
import cPickle
import collections
import glob
import json
import numpy as np
import os
import random
import re
import sys
import threading
import time

seed = int(time.time())
#seed = 0
random.seed(seed)
np.random.seed(seed)

def copy_to_gpu(data):
  return gpuarray.to_gpu(data.astype(np.float32))

def consistent_shuffle(lst):
  '''Shuffle a list in a way that doesn't change between invocations.'''
  random.seed(len(lst))
  np.random.seed(len(lst))
  np.random.shuffle(lst)
  return lst


class BatchData(object):
  def __init__(self, data, labels, epoch):
    self.data = data
    self.labels = labels
    self.epoch = epoch


class DataProvider(object):
  def __init__(self, data_dir='.', batch_range=None):
    self.data_dir = data_dir
    self.meta_file = os.path.join(data_dir, 'batches.meta')

    self.curr_batch_index = 0
    self.curr_batch = None
    self.curr_epoch = 1

    if os.path.exists(self.meta_file):
      self.batch_meta = util.load(self.meta_file)
    else:
      util.log_warn('Missing metadata for loader.')

    if batch_range is None:
      self.batch_range = self.get_batch_indexes()
    else:
      self.batch_range = batch_range

    util.log('Batch range: %s', self.batch_range)
    random.shuffle(self.batch_range)

    self.index = 0
    self._handle_new_epoch()

  def reset(self):
    self.curr_batch_index = 0
    self.curr_batch = None
    self.curr_epoch = 1
    random.shuffle(self.batch_range)

  def get_next_index(self):
    self.curr_batch_index = self.curr_batch_index + 1
    if self.curr_batch_index == len(self.batch_range) + 1:
      random.shuffle(self.batch_range)
      self.curr_epoch += 1
      self.curr_batch_index = 1

      self._handle_new_epoch()

    self.curr_batch = self.batch_range[self.curr_batch_index - 1]
    return self.curr_batch

  def _handle_new_epoch(self):
    '''
    Called when a new epoch starts.
    '''
    pass

  def get_batch_num(self):
    return len(self.batch_range)

  @property
  def image_shape(self):
    return (3, self.inner_size, self.inner_size)

  @property
  def data_dim(self):
    return self.inner_size ** 2 * 3

  def _trim_borders(self, images, target):
    if self.multiview:
      start_positions = [(0, 0), (0, self.border_size * 2), (self.border_size, self.border_size),
                         (self.border_size *2 , 0), (self.border_size * 2 , self.border_size * 2)]
      end_positions = [(x + self.inner_size, y + self.inner_size) for (x, y) in start_positions]
      for i in xrange(self.num_view / 2):
        startY , startX = start_positions[i][0], start_positions[i][1]
        endY, endX = end_positions[i][0], end_positions[i][1]
        num_image = len(images)
        for idx, img in enumerate(images):
          pic = img[:, startY:endY, startX:endX]
          target[:, i * num_image + idx] = pic.reshape((self.data_dim, ))
          target[:, (self.num_view/2 +i) * num_image + idx] = pic[:, :, ::-1].reshape((self.data_dim, ))
    else:
      for idx, img in enumerate(images):
        startY, startX = np.random.randint(0, self.border_size * 2 + 1), np.random.randint(0, self.border_size * 2 + 1)
        #startY, startX = 0, 0
        endY, endX = startY + self.inner_size, startX + self.inner_size
        pic = img[:, startY:endY, startX:endX]
        #if False:
        if np.random.randint(2) == 0:  # also flip the image with 50% probability
          pic = pic[:, :, ::-1]
     
        #print pic.shape, target[:,idx].shape
        target[:, idx] = pic.reshape((self.data_dim,))


def _prepare_images(data_dir, category_range, batch_range, batch_meta):
  assert os.path.exists(data_dir), data_dir

  dirs = glob.glob(data_dir + '/n*')
  synid_to_dir = {}
  for d in dirs:
    synid_to_dir[basename(d)[1:]] = d

  if category_range is None:
    cat_dirs = dirs
  else:
    cat_dirs = []
    for i in category_range:
      synid = batch_meta['label_to_synid'][i]
      # util.log('Using category: %d, synid: %s, label: %s', i, synid, self.batch_meta['label_names'][i])
      cat_dirs.append(synid_to_dir[synid])

  images = []
  batch_dict = dict((k, k) for k in batch_range)

  for d in cat_dirs:
    imgs = [v for i, v in enumerate(glob.glob(d + '/*.jpg')) if i in batch_dict]
    images.extend(imgs)

  return np.array(images)


class ImageNetDataProvider(DataProvider):
  img_size = 256
  border_size = 16
  inner_size = 224

  def __init__(self, data_dir,
               batch_range=None,
               multiview = False,
               category_range=None,
               scale=1,
               batch_size=1024):
    DataProvider.__init__(self, data_dir, batch_range)
    self.multiview = multiview
    self.batch_size = batch_size

    self.scale = scale

    self.img_size = ImageNetDataProvider.img_size / scale
    self.border_size = ImageNetDataProvider.border_size / scale
    self.inner_size = self.img_size - self.border_size * 2

    if self.multiview:
      self.batch_size = 12

    self.images = _prepare_images(data_dir, category_range, batch_range, self.batch_meta)
    self.num_view = 5 * 2 if self.multiview else 1

    assert len(self.images) > 0

    self._shuffle_batches()

    if 'data_mean' in self.batch_meta:
      data_mean = self.batch_meta['data_mean']
    else:
      data_mean = util.load(data_dir + 'image-mean.pickle')['data']

    self.data_mean = (data_mean
        .astype(np.single)
        .T
        .reshape((3, 256, 256))[:,
                                self.border_size:self.border_size + self.inner_size,
                                self.border_size:self.border_size + self.inner_size]
        .reshape((self.data_dim, 1)))
    util.log('Starting data provider with %d batches', len(self.batches))

  def _shuffle_batches(self):
    # build index vector into 'images' and split into groups of batch-size
    image_index = np.arange(len(self.images))
    np.random.shuffle(image_index)

    self.batches = np.array_split(image_index,
                                  util.divup(len(self.images), self.batch_size))

    self.batch_range = range(len(self.batches))
    np.random.shuffle(self.batch_range)

  def _handle_new_epoch(self):
    self._shuffle_batches()

  def get_next_batch(self):
    self.get_next_index()

    epoch = self.curr_epoch
    batchnum = self.curr_batch
    names = self.images[self.batches[batchnum]]
    num_imgs = len(names)
    labels = np.zeros((1, num_imgs))
    cropped = np.ndarray((self.data_dim, num_imgs * self.num_view), dtype=np.uint8)
    # _load in parallel for training
    st = time.time()
    images = []
    for idx, filename in enumerate(names):
#       util.log('Loading... %s %s', idx, filename)
      jpeg = Image.open(filename)
      if jpeg.mode != "RGB": jpeg = jpeg.convert("RGB")
      if self.scale != 1:
        x, y = jpeg.size
        jpeg = jpeg.resize((x / self.scale, y / self.scale), Image.LINEAR)

      # starts as rows * cols * rgb, tranpose to rgb * rows * cols
      img = np.asarray(jpeg, np.uint8).transpose(2, 0, 1)
      images.append(img)

    self._trim_borders(images, cropped)
    #if self.test:
    #  np.set_printoptions(threshold = np.nan)
    #  print cropped[:, 0]
    #  print cropped[:, num_imgs]

    load_time = time.time() - st

    clabel = []
    # extract label from the filename
    for idx, filename in enumerate(names):
      filename = os.path.basename(filename)
      synid = filename[1:].split('_')[0]
      label = self.batch_meta['synid_to_label'][synid]
      labels[0, idx] = label

    st = time.time()
    cropped = cropped.astype(np.single)
    cropped = np.require(cropped, dtype=np.single, requirements='C')
    cropped -= self.data_mean

    align_time = time.time() - st

    labels = np.array(labels)
    labels = labels.reshape(labels.size,)
    labels = np.require(labels, dtype=np.single, requirements='C')

    # util.log("Loaded %d images in %.2f seconds (%.2f _load, %.2f align)",
    #         num_imgs, time.time() - start, load_time, align_time)
    # self.data = {'data' : SharedArray(cropped), 'labels' : SharedArray(labels)}

    return BatchData(cropped, labels, epoch)


class CifarDataProvider(DataProvider):
  img_size = 32
  border_size = 0
  inner_size = 32

  BATCH_REGEX = re.compile('^data_batch_(\d+)$')
  def get_next_batch(self):
    self.get_next_index()
    filename = os.path.join(self.data_dir, 'data_batch_%d' % self.curr_batch)

    data = util.load(filename)
    img = data['data'] - self.batch_meta['data_mean']
    return BatchData(np.require(img, requirements='C', dtype=np.float32),
                     np.array(data['labels']),
                     self.curr_epoch)

  def get_batch_indexes(self):
    names = self.get_batch_filenames()
    return sorted(list(set(int(DataProvider.BATCH_REGEX.match(n).group(1)) for n in names)))

class CroppedCifarDataProvider(CifarDataProvider):
  inner_size = 24
  img_size = 32
  border_size = 4
  multiview = False


  def __init__(self, data_dir = '.', batch_range = None, multiview = False):
    CifarDataProvider.__init__(self, data_dir, batch_range)
    self.multiview = multiview
    if self.multiview:
      self.num_view = 10
    else:
      self.num_view = 1
    self.batch_meta['data_mean'] = self.batch_meta['data_mean'].reshape((3, 32, 32))
    self.batch_meta['data_mean'] = self.batch_meta['data_mean'][:, self.border_size:self.border_size + self.inner_size, self.border_size:self.border_size + self.inner_size]
    self.batch_meta['data_mean'] = self.batch_meta['data_mean'].reshape((self.data_dim, 1))

  def get_next_batch(self):
    self.get_next_index()
    filename = os.path.join(self.data_dir, 'data_batch_%d' % self.curr_batch)

    data = util.load(filename)
    img = data['data']
    img = img.reshape((3, 32, 32, len(data['labels'])))
    img = img.transpose(3, 0, 1, 2)
    cropped = np.ndarray((self.data_dim, len(data['labels']) * self.num_view), dtype=np.float32)

    self._trim_borders(img, cropped)
    cropped -= self.batch_meta['data_mean']

    return BatchData(np.require(cropped, requirements='C', dtype=np.float32),
                     np.array(data['labels']),
                     self.curr_epoch)



class IntermediateDataProvider(DataProvider):
  def __init__(self, data_dir, batch_range, data_name):
    DataProvider.__init__(self, data_dir, batch_range)
    self.data_name = data_name

  def get_next_batch(self):
    self.get_next_index()

    filename = os.path.join(self.data_dir + '.%s' % self.curr_batch)
    util.log('reading from %s', filename)

    data_dic = util.load(filename)
    data  = data_dic[self.data_name].transpose()
    labels = data_dic['labels']
    data = np.require(data, requirements='C', dtype=np.float32)
    return BatchData(data, labels, self.curr_epoch)



class MemoryDataProvider(DataProvider):
  def __init__(self, data_holder, batch_range = None, data_name = 'fc'):
    data_holder.finish_push()
    if batch_range is None:
      batch_range  = range(data_holder.get_count())

    DataProvider.__init__(self, data_dir = '.', batch_range = batch_range)
    self.data_holder = data_holder
    self.data_list = self.data_holder.memory_chunk
    self.data_name = data_name

  def get_next_batch(self):
    self.get_next_index()

    data = self.data_list[self.curr_batch]
    labels = data['labels']
    img = np.require(data[self.data_name].transpose(), requirements='C', dtype=np.float32)
    return BatchData(img, labels, self.curr_epoch)

class SimpleDataProvider(DataProvider):
  multiview = False
  num_view = 1

  def __init__(self, data_dir, target_size, 
               batch_size=16,
               crop=0, start_pos=0.0, end_pos=1.0, **kw):
    self.data_dir = data_dir
    self.dp_init()
    self.target_size = target_size
    self.crop = crop

    self.border_size = self.crop
    self.inner_size = self.target_size - self.crop * 2

    self.input_range = range(int(self.num_inputs * start_pos),
                             int(self.num_inputs * end_pos))
    self.batch_size = batch_size

    DataProvider.__init__(self, data_dir, batch_range=range(0, len(self.input_range) / self.batch_size))


  def get_next_batch(self):
    idx = self.get_next_index()
    sz = self.target_size
    labels = []
    indices = self.input_range[idx * self.batch_size:(idx + 1) * self.batch_size]
    images = []

    for i, load_idx in enumerate(indices):
      image, label = self.load(load_idx)
      image = image.resize((sz, sz))
      image = image.convert('RGB')
      images.append(np.asarray(image).transpose(2, 0, 1))
      labels.append(label)

    if self.crop > 0:
      print self.data_dim, len(images)
      cropped = np.ndarray((self.data_dim, len(images) * self.num_view), dtype=np.uint8)
      self._trim_borders(images, cropped)
      return BatchData(cropped, labels, self.curr_epoch)

    return BatchData(images, labels, self.curr_epoch)


class DirectoryDataProvider(SimpleDataProvider):
  def dp_init(self):
    files = glob.glob(self.data_dir + '/*/*')
    image_files = []
    for f in files:
      try:
        Image.open(f)
        image_files.append(f)
      except:
        pass

    assert len(image_files) > 0, 'No image files found in %s' % self.data_dir
    print 'Found %d files' % len(image_files)
    
    label_file = self.data_dir + '/LABELS.json'
    try:
      self.label_dict = json.load(open(label_file))
    except:
      print 'No label file found, will create one.'
      self.label_dict = {}

    for f in image_files:
      type = f.split('/')[-2]
      if not type in self.label_dict:
        self.label_dict[type] = len(self.label_dict)

    with open(label_file, 'w') as l:
      json.dump(self.label_dict, l)

    self.image_files = consistent_shuffle(image_files)
    self.num_inputs = len(self.image_files)

  def load(self, idx):
    image_file = self.image_files[idx]
    type = image_file.split('/')[-2]
    label = self.label_dict[type]
    return (Image.open(self.image_files[idx]), label)



class ReaderThread(threading.Thread):
  def __init__(self, queue, dp):
    threading.Thread.__init__(self)
    self.daemon = True
    self.queue = queue
    self.dp = dp
    self._stop = False
    self._running = True

  def run(self):
    while not self._stop:
      #util.log('Fetching...')
      self.queue.put(self.dp.get_next_batch())
      #util.log('%s', self.dp.curr_batch_index)
      #util.log('Done.')

    self._running = False

  def stop(self):
    self._stop = True
    while self._running:
      _ = self.queue.get(0.1)

try:
  from pycuda import gpuarray
  from fastnet.cuda_kernel import gpu_partial_copy_to
except:
  print 'PyCuda not found, disabling parallel dataprovider.'
else:
  class ParallelDataProvider(DataProvider):
    def __init__(self, dp):
      self.dp = dp
      self._reader = None
      self.reset()

    def _start_read(self):
      util.log('Starting reader...')
      assert self._reader is None
      self._reader = ReaderThread(self._data_queue, self.dp)
      self._reader.start()

    @property
    def image_shape(self):
      return self.dp.image_shape

    @property
    def multiview(self):
      if hasattr(self.dp, 'multiview'):
        return self.dp.multiview
      else:
        return False

    @property
    def batch_size(self):
      if hasattr(self.dp, 'batch_size'):
        return self.dp.batch_size
      else:
        return 0

    @property
    def num_view(self):
      if hasattr(self.dp, 'num_view'):
        return self.dp.num_view
      else:
        return 1

    def reset(self):
      self.dp.reset()

      if self._reader is not None:
        self._reader.stop()

      self._reader = None
      self._data_queue = Queue.Queue(1)
      self._gpu_batch = None
      self.index = 0

    def _fill_reserved_data(self):
      batch_data = self._data_queue.get()

      #timer = util.EZTimer('fill reserved data')

      self.curr_epoch = batch_data.epoch
      if not self.multiview:
        batch_data.data = copy_to_gpu(batch_data.data)
        batch_data.labels = copy_to_gpu(batch_data.labels)
        self._gpu_batch = batch_data
      else:
        self._cpu_batch = batch_data

    def get_next_batch(self, batch_size):
      if self._reader is None:
        self._start_read()

      if self._gpu_batch is None:
        self._fill_reserved_data()

      if not self.multiview:
        height, width = self._gpu_batch.data.shape
        gpu_data = self._gpu_batch.data
        gpu_labels = self._gpu_batch.labels
        epoch = self._gpu_batch.epoch

        if self.index + batch_size >=  width:
          width = width - self.index
          labels = gpu_labels[self.index:self.index + batch_size]

          data = gpuarray.zeros((height, width), dtype = np.float32)
          gpu_partial_copy_to(gpu_data, data, 0, height, self.index, self.index + width)
          self.index = 0
          self._fill_reserved_data()
        else:
          labels = gpu_labels[self.index:self.index + batch_size]
          data = gpuarray.zeros((height, batch_size), dtype = np.float32)
          gpu_partial_copy_to(gpu_data, data, 0, height, self.index, self.index + batch_size)
          self.index += batch_size
      else:
        # multiview provider
        # number of views should be 10
        # when using multiview, do not pre-move data and labels to gpu
        height, width = self._cpu_batch.data.shape
        cpu_data = self._cpu_batch.data
        cpu_labels = self._cpu_batch.labels
        epoch = self._cpu_batch.epoch

        width /= self.num_view

        if self.index + batch_size >=  width:
          batch_size = width - self.index

        labels = cpu_labels[self.index:self.index + batch_size]
        data = np.zeros((height, batch_size * self.num_view), dtype = np.float32)
        for i in range(self.num_view):
          data[:, i* batch_size: (i+ 1) * batch_size] = cpu_data[:, self.index + width * i : self.index + width * i + batch_size]

        data = copy_to_gpu(np.require(data, requirements = 'C'))
        labels = copy_to_gpu(np.require(labels, requirements = 'C'))


        self.index = (self.index + batch_size) / width
      
      #util.log_info('Batch: %s %s %s', data.shape, gpu_labels.shape, labels.shape)
      return BatchData(data, labels, epoch)


dp_dict = {}
def register_data_provider(name, _class):
  assert not name in dp_dict, ('Data Provider', name, 'already registered')
  dp_dict[name] = _class

def get_by_name(name):
  assert name in dp_dict, 'No such data provider %s' %  name
  dp_klass = dp_dict[name]
  def construct_dp(*args, **kw):
    dp = dp_klass(*args, **kw)
    return ParallelDataProvider(dp)
  return construct_dp

def from_class(dp_klass):
  def construct_dp(*args, **kw):
    dp = dp_klass(*args, **kw)
    return ParallelDataProvider(dp)
  return construct_dp


register_data_provider('cifar10', CifarDataProvider)
register_data_provider('cropped-cifar10', CroppedCifarDataProvider)
register_data_provider('imagenet', ImageNetDataProvider)
register_data_provider('intermediate', IntermediateDataProvider)
register_data_provider('memory', MemoryDataProvider)
