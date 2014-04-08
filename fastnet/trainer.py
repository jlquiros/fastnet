import sys
import time

from traits.api import HasTraits, Int, Float, Any, Bool, Instance, List

from fastnet.checkpoint import CheckpointDumper
from fastnet.layer import TEST

from . import util


class Trainer(HasTraits):
  train_dp = Any()
  test_dp = Any()
  network = Any()
  batch_size = Int()
  multiview = Bool()
  test_freq = Int(100)
  save_freq = Int(1000)
  checkpointer = Instance(CheckpointDumper, allow_none=False)
  
  curr_epoch, curr_batch = Int(), Int()
  start_time = Float()
  base_time = Float() 
  
  train_outputs = List()
  test_outputs = List()

  def __init__(self, *args, **kw):
    HasTraits.__init__(self, *args, **kw)
    self.start_time = time.time()
    self.curr_epoch = 0
    self.curr_batch = 0

    self.train_outputs = []
    self.test_outputs = []
    self.base_time = 0

    self._finish_init()

  def _finish_init(self):
    pass

  def init_data_provider(self):
    self.train_dp.reset()
    self.test_dp.reset()

  def checkpoint(self):
    model = {}
    model['net'] = self.network
    model['train_outputs'] = self.train_outputs
    model['test_outputs'] = self.test_outputs

    print >> sys.stderr, '---- save checkpoint ----'
    self.print_net_summary()
    self.checkpointer.dump(checkpoint=model, suffix=self.curr_epoch)


  def adjust_lr(self):
    print >> sys.stderr, '---- adjust learning rate ----'
    self.network.adjust_learning_rate(self.factor)

  def elapsed(self):
    return time.time() - self.start_time + self.base_time

  def test(self):
    batch_size = self.batch_size
    test_data = self.test_dp.get_next_batch(batch_size)

    input, label = test_data.data, test_data.labels
    if self.multiview:
      num_view = self.test_dp.num_view
      self.network.test_batch_multiview(input, label, num_view)
    else:
      self.network.train_batch(input, label, TEST)

    cost, correct = self.network.get_batch_information()
    num_examples = batch_size

    self.test_outputs += [({'logprob': [cost, 1 - correct]}, num_examples, self.elapsed())]
    print >> sys.stderr, '---- test ----'
    print >> sys.stderr, 'error: %f logreg: %f' % (1 - correct, cost)

    self.network.check_weights()

  def print_net_summary(self):
    print >> sys.stderr, '--------------------------------------------------------------'
    for s in self.network.get_summary():
      name = s[0]
      values = s[1]
      print >> sys.stderr, "Layer '%s' weight: %e [%e] @ [%e]" % (name, values[0], values[1],
          values[4])
      print >> sys.stderr, "Layer '%s' bias: %e [%e] @ [%e]" % (name, values[2], values[3],
          values[5])


  def should_test(self):
    return self.curr_batch % self.test_freq == 0

  def should_checkpoint(self):
    return self.curr_batch % self.save_freq == 0

  def _finished_training(self):
    pass

  def train(self, num_batches):
    self.print_net_summary()
    util.log('Starting training...')

    start_epoch = self.curr_epoch
    last_print_time = time.time()

    for i in xrange(num_batches):
      batch_start = time.time()
      train_data = self.train_dp.get_next_batch(self.batch_size)
      num_examples = train_data.data.shape[-1]
      self.curr_epoch = train_data.epoch
      self.curr_batch += 1

      input, label = train_data.data, train_data.labels
      self.network.train_batch(input, label)
      cost, correct = self.network.get_batch_information()
      self.train_outputs += [({'logprob': [cost, 1 - correct]}, num_examples, self.elapsed())]

      if time.time() - last_print_time > 1:
        print >> sys.stderr, '%d.%d: error: %f logreg: %f time: %f' % (
                      self.curr_epoch, 
                      self.curr_batch, 
                      1 - correct, 
                      cost, 
                      time.time() - batch_start)
        last_print_time = time.time()

      if self.should_checkpoint():
        self.checkpoint()

      if self.should_test():
        self.test()

    self.test()
    self.checkpoint()
    self._finished_training()

