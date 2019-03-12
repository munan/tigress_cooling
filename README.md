Heating and cooling functions for TIGRESS simulation.

Implementation is in tigress_cool.h and tigress_cool.c
Please check the TODO comments in these two files for integrating it with
Athena-TIGRESS code. Original implementation by Munan Gong. 

Linecooling functions are from photoionization code [CMacIonize](https://github.com/bwvdnbro/CMacIonize).

Python wrappers written by Jeong-Gyu Kim.

Run

`$ make lib`

to create shared library files.

Python package [aenum](https://pypi.org/project/aenum/) is required.
