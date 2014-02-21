#!/usr/bin/python

from PIL import Image
from synids import SYNID_NAMES, SYNID_TO_LABEL
import cPickle
import cStringIO
import struct
import sys
import zipfile
import numpy as N

IMAGESIZE = 384

def write_str(s):
  sys.stdout.write(struct.pack('>i', len(s)))
  sys.stdout.write(s)
  
def read_str():
  packed_len = sys.stdin.read(4)
  if not packed_len or len(packed_len) < 4: return None
  strlen = struct.unpack('>i', packed_len)[0]
  return sys.stdin.read(strlen)

def write_kv(k, v):
  write_str(k)
  write_str(v)
  
def read_kv():
  while 1:
    k = read_str()
    v = read_str()
    if k is None or v is None:
      break
    yield (k, v)

def resize_and_crop(data):
  data_file = cStringIO.StringIO(data)
  img = Image.open(data_file).convert('RGB')
  out = Image.new('RGB', (IMAGESIZE, IMAGESIZE))
  
  thumb = img.copy()
  thumb.thumbnail((IMAGESIZE, IMAGESIZE), Image.BICUBIC)

  x_off = (IMAGESIZE - thumb.size[0]) / 2
  y_off = (IMAGESIZE - thumb.size[1]) / 2
  box = (x_off, y_off, x_off + thumb.size[0], y_off + thumb.size[1])
  out.paste(thumb, box)

  bytes_out = cStringIO.StringIO()
  out.save(bytes_out, 'jpeg', quality=90)
  return bytes_out.getvalue()

def resize_and_crop_mapper():
  for filename, data in read_kv():
    blob = resize_and_crop(data)

    # write out the hash of the filename, so that when the reducer
    # sorts inputs, we don't end up with each class being next to
    # each other and bungling the training.
    write_kv(str(hash(filename)), blob)

def compute_mean_mapper():
  total = N.zeros((IMAGESIZE, IMAGESIZE, 3), dtype=N.double)
  count = 0

  for filename, data in read_kv():
    print >>sys.stderr, 'Working... %s' % filename  
    imginfo = cPickle.loads(data)
    img = N.asarray(Image.open(cStringIO.StringIO(imginfo['data'])), dtype=N.double)
    total += img
    count += 1

  write_kv('image-mean', cPickle.dumps((total, count), protocol=-1))

def compute_mean_reducer():
  total = N.zeros((IMAGESIZE, IMAGESIZE, 3), dtype=N.double)
  count = 0
  for key, value in read_kv():
    print >>sys.stderr, 'Processing... '
    mtotal, mcount = cPickle.loads(value)
    count += mcount
    total += mtotal
  
  output = {}
  output['mean'] = total / count
  output['num_images'] = count
  output['total_value'] = total
  write_kv('image-mean', cPickle.dumps(output, protocol=-1))
 
if __name__ == '__main__':
  op = sys.argv[1]
  if not op in globals():
    print  >>sys.stderr, 'Bad command line:', sys.argv
    sys.exit(1)
  globals()[op]()
