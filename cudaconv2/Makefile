SRC = cudaconv2_wrap.cu filter_acts.cu img_acts.cu weight_acts.cu conv_util.cu 
OBJ = $(patsubst %.cu,%.o,${SRC})

NVCC_OPTS := --pre-include undef.h
CXX_FLAGS := --compiler-options='-fPIC -ggdb2'
PYTHON_INCLUDE := /usr/include/python2.7
INCLUDES := -I${PYTHON_INCLUDE} -I. -I./cutil

all: _cudaconv2.so

cudaconv2_wrap.cu: cudaconv2.i
	swig -python -threads -c++ -o $@ ${INCLUDES} $<

%.o : %.cu 
	nvcc -arch=sm_20 -O3 -g -c -o $@ $<  ${NVCC_OPTS} ${CXX_FLAGS} ${INCLUDES}

_cudaconv2.so: ${OBJ}
	nvcc -shared -o $@ $^

clean:
	rm -f *.o *.so
