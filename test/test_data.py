from fastnet import data, util
from fastnet.util import print_matrix
from PIL import Image
import numpy as np
import time
import os

def test_imagenet_loader():
  df = data.get_by_name('imagenet')(
                                 '/ssd/nn-data/imagenet/', 
                                 batch_range=range(1000), 
                                 test = True,
                                 batch_size=128)
  
  for i in range(32):
    st = time.time()
    batch = df.get_next_batch(8 * 10)
    print time.time() - st
    print batch.labels
    print batch.data.shape
    time.sleep(0.5)
  print batch.labels


def test_cifar_loader():
  data_dir = '/ssd/nn-data/cifar-10.old/'
  dp = data.get_by_name('cifar10')(data_dir, [1])
  batch_size = 128
  
  data_list = []
  for i in range(11000):
    batch = dp.get_next_batch(batch_size)
    batch = batch.data.get()
    data_list.append(batch)

    if batch.shape[1] != batch_size:
      break
  batch = np.concatenate(data_list, axis = 1)
  print_matrix(batch, 'batch')

def test_dir_loader():
  os.system('mkdir -p /tmp/imgdir')
  for i in range(10):
    os.system('mkdir -p /tmp/imgdir/cat-%d' % i)
    for j in range(10):
      Image.new('RGB', (256, 256), 'white').save('/tmp/imgdir/cat-%d/%d.jpg' % (i, j))

  data_dir = '/tmp/imgdir'
  dp = data.DirectoryDataProvider(data_dir=data_dir, crop=1, target_size=32)
  batch = dp.get_next_batch()
  print batch.labels
