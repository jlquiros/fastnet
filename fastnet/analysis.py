#!/usr/bin/env

'''Functions for analyzing the output of fastnet checkpoint files.'''

import numpy as np

from matplotlib.pyplot import gcf
import matplotlib.pyplot as plt
import pylab

from . import util


def plot_df(df, x, y, save_to=None, title=None, merge=False,
            x_label=None, y_label=None, legend=None,
            transform_x=lambda k, x: x, transform_y=lambda k, y: y,
            xlim=None, ylim=None):
  from itertools import cycle

  lines = cycle(["-", "--", "-.", ":"])
  colors = cycle('bgrcmyk')

  if merge:
    f = gcf()
  else:
    f = plt.figure()

  if isinstance(df, dict):
    for k in sorted(df.keys()):
      v = df[k]
      ax = f.add_subplot(111)
      ax.plot(transform_x(k, v[x]), transform_y(k, v[y]),
              linestyle=lines.next(), color=colors.next(), label='%s' % k)
  else:
    ax = f.add_subplot(111)
    ax.plot(df[x], df[y], linestyle=lines.next(), color=colors.next())

  ax.set_title(title)
  if legend: ax.legend(title=legend)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  if xlim is not None: ax.set_xlim(xlim)
  if ylim is not None: ax.set_ylim(ylim)
  #ax.set_yscale('log')

  f.set_figheight(8)
  f.set_figwidth(12)
  if save_to is not None:
    pylab.savefig(save_to, bbox_inches=0)


def plot_series(frame, groupby, x, y, **kw):
  g = frame.groupby(groupby)
  df = dict([(k, g.get_group(k)) for k in g.groups.keys()])
  kw['x_label'] = x
  kw['y_label'] = y
  plot_df(df, x, y, **kw)


def build_image(array):
  if len(array.shape) == 4:
    filter_size = array.shape[1]
  else:
    filter_size = array.shape[0]

  num_filters = array.shape[-1]
  num_cols = util.divup(80, filter_size)
  num_rows = util.divup(num_filters, num_cols)

  if len(array.shape) == 4:
    big_pic = np.zeros(
      (3, (filter_size + 1) * num_rows, (filter_size + 1) * num_cols))
  else:
    big_pic = np.zeros((filter_size * num_rows, filter_size * num_cols))

  for i in range(num_rows):
    for j in range(num_cols):
      idx = i * num_cols + j
      if idx >= num_filters: break
      x = i * (filter_size + 1)
      y = j * (filter_size + 1)
      if len(array.shape) == 4:
        big_pic[:, x:x + filter_size, y:y + filter_size] = array[:, :, :, idx]
      else:
        big_pic[x:x + filter_size, y:y + filter_size] = array[:, :, idx]

  if len(array.shape) == 4:
    return big_pic.transpose(1, 2, 0)
  return big_pic


def plot_filters(imgs, ax=None):
  imgs = imgs - imgs.min()
  imgs = imgs / imgs.max()

  if ax is None:
    fig = pylab.gcf()
    fig.set_size_inches(12, 8)
    ax = fig.add_subplot(111)

  big_pic = build_image(imgs)
  ax.imshow(big_pic, interpolation='nearest')


