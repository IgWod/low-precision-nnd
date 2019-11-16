# Introduction

The following code was used to obtain results for the paper: "Low-Precision Neural
Network Decoding of Polar Codes" published to SPAWC2019. If you use this code please cite us:

```
@inproceedings{wodiany2019low,
  title={{Low-Precision} Neural Network Decoding of Polar Codes},
  author={Wodiany, Igor and Pop, Antoniu},
  booktitle={2019 IEEE 20th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC)},
  pages={1--5},
  year={2019},
  organization={IEEE}
}
```

# Running Python simulation

nn_decoder.py - layer type
run.py get_decoder - NN sizes
layers.py - quantizations levels

Running the simulation consists of 2 steps: Setting up the environment and running
the python function. The first step is achieved by following commands:

```bash
$ cd <Project Root>/python
$ pipenv shell
$ python3
```

Then to run the example simulation:

```
>>> import simulation.run
>>> simulation.run.run_simulation(8, 4, ['NN_LLR'], 1000)
```

Python simulation supports caching now so when model is invoked again, after it was
trained in the previous run, simulation will loaded it from the saved file. However it does not
work for the NN with quantized layers.

The following files have to be changed to test different configurations:

* Layers type (normal/quantized) can be changed in *nn_decoder.py*
* Layers size can be changed in *run.py* in the *get_decoder* function
* Quantizated types can be changed in *layers.py*

# Running C++ implementation

The C++ project can be built using CMake and Make using the Intel Compiler (ICC):

```
$ mkdir build
$ cd build/
$ cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/intel/compilers_and_libraries_2019/linux/bin/intel64/icc \
  -DCMAKE_CXX_COMPILER=/opt/intel/compilers_and_libraries_2019/linux/bin/intel64/icc ..
$ make
```

The project was tested with ICC 19.0.1, however it should work with any recent version of ICC or GCC.

To run the decoder:

```
$ cd source/polar_decoder
$ ./Decoder 512-256-128/int8
```

The second argument is the network configuration and the date type. Only configurations that are
in the data/weights directory are valid arguments.

The following changes have to be made to the source code to allow testing different sizes/data types:

* Adjust layers sizes in *NeuralDecoder.h*
* Change weights data types in *DecoderMain.cpp*
* Enable or disable lut2 clipping in *DenseLayer.h*

# Questions

Since the code is not the highest quality, as getting results fast was far more important
than everything else, please feel to contact me if you need help with running the code. 
