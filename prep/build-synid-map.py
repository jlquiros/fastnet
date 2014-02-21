#!/usr/bin/env python

import glob, re, cPickle
from os.path import basename
DATADIR = '/hdfs/imagenet/'

def synids():
  ids = glob.glob(DATADIR + '/zip/*.zip')
  ids = [basename(x)[1:-4] for x in ids]
  return ids

def synid_to_name():
  syns = open(DATADIR + '/fall11_synsets.txt').read().split('\n')
  syns = dict([re.split(' ', x, maxsplit=1) for x in syns][:-1])
  for k, v in syns.items():
    syns[k] = v.split(',')[0]
  return syns
 
SYNIDS = synids()
SYNID_NAMES = synid_to_name()
LABEL_NAMES = [SYNID_NAMES[s] for s in SYNIDS]

with open('./synids.py', 'w') as f:
  print >>f, 'SYNIDS = ['
  for idx, syn in enumerate(SYNIDS):
    print >>f, '[%d, "%s", "%s"],' % (idx, syn, SYNID_NAMES[syn])
  print >>f, ']',
  print >>f, '''
SYNID_NAMES = {}
SYNID_TO_LABEL = {}
LABEL_NAMES = []

for label_idx, synid, name in SYNIDS:
  SYNID_NAMES[synid] = name
  SYNID_TO_LABEL[synid] = label_idx
  LABEL_NAMES.append(name)
''' 

# output batches.meta

meta = {}
meta['label_names'] = LABEL_NAMES
with open('./batches.meta', 'w') as m:
  m.write(cPickle.dumps(meta))

