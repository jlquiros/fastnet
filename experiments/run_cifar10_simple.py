#!/usr/bin/python2.7

from fastnet import data, trainer, net, parser
from pycuda import autoinit

test_id = 'cifar-test-1'

data_dir = '/ssd0/cifar-10-python/'
checkpoint_dir = './checkpoint/'
param_file = './config/cifar-10-18pct.cfg'

train_range = range(1, 6) #1,2,3,....,40
test_range = range(6, 7) #41, 42, ..., 48
data_provider = 'cropped-cifar10'


multiview = False
train_dp = data.get_by_name(data_provider)(data_dir,train_range)
test_dp = data.get_by_name(data_provider)(data_dir, test_range, multiview = multiview)
checkpointer = trainer.CheckpointDumper(checkpoint_dir, test_id)

init_model = checkpointer.get_checkpoint()
if init_model is None:
  init_model = parser.parse_config_file(param_file)

save_freq = 100
test_freq = 100
adjust_freq = 100
factor = 1
num_epoch = 10
learning_rate = 0.01
batch_size = 64
image_color = 3
image_size = 24
image_shape = (image_color, image_size, image_size, batch_size)

network = parser.load_model(net.FastNet(image_shape), init_model)

param_dict = globals()
t = trainer.Trainer(**param_dict)
t.train(10000)
t.predict()
