Fastnet
=========
A [convolutional neural network](http://yann.lecun.com/exdb/lenet/) framework, building on 
top of the convolutional kernel code from [cuda-convnet](https://code.google.com/p/cuda-convnet/).


**Setup**

```
git clone https://github.com/rjpower/fastnet
cd fastnet
python setup.py develop [--user]
```

**Usage**

To run a trainer directly:

    python fastnet/trainer.py --help
    
Take a look at the scripts in `fastnet/scripts` for examples of how to run your own network.


**Requires**

  * [NumPy](http://www.numpy.org/)
  * [CUDA](http://www.nvidia.com/object/cuda_home_new.html)
  * [PyCUDA](http://documen.tician.de/pycuda/)

**Referencing**

If you use FastNet for your research, please cite us in your publication:

R. Power and J. Lin.<br>
*FastNet -- A framework for convolutional neural networks*.<br>
_https://github.com/rjpower/fastnet_

Bibtex:

    @misc{PowerFastNet,
          author = {Power, Russell and Lin, Justin},
          title = {{F}ast{N}et -- {A} framework for convolutional neural networks},
          howpublished = {\url{https://github.com/rjpower/fastnet}},
    }
