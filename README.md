# perl_machine_learning
Collection of Perl scripts and packages implementing various Machine Learning/AI strategies 

Developed and tested on Debian.

# Prerequisites

* CPAN modules from Debian repositories - `apt install pdl libfile-slurp-perl libjson-perl libmath-random-perl libmodern-perl-perl`
* Other CPAN modules - `cpan install Math::Matrix`
* Create a directory called MNIST and download and uncompress the MNIST data - download from [MNIST repository](http://yann.lecun.com/exdb/mnist/)

# 1 Hidden Layer Neural Networks

## nn.py
Code from [Neural Networks and Deep Learning - Chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html) - this is a python script, included only for reference.  Modified from the original to output its initialisation weights and biases, and mini batches to JSON.  At the time of writing, the Pickle modules in CPAN couldn't read python3 pkl files, so the data is stored in JSON files.

## nn_1.pl
A Perl version of nn.py.  Painfully slow, taking 40 minutes per Epoch.  Same accuracy as nn.py, at around 95%.  After installing the prerequistes, run with `perl nn_1.pl`.  After running nn.py all of the JSON files will have been created, and you can use exactly the same weights and biases, and mini batches by updating 'load_from_file' to 1 in nn_1.pl.  Checking the intermediate matricies should show exactly the same values at the same stage as nn.py.

## nn_1_pdl.pl
Updated from nn_1.pl to use PDL for Matrix manipulation.  About twice as fast as nn_1.pl, taking about 20 minutes per epoch.  95% accuracy.  Run with `perl nn_1_pdl.pl`.

## nn_1_cuda_c.pl
Updated from nn_1.pl to use CUDA for Matrix manipulation.  Takes 6 **seconds** per epoch.  95% accuracy.  Run with `perl nn_1_cuda_c.pl`.  Requires the Perl module Inline::CUDA (and all of its prerequistites - see [Inline::CUDA](https://github.com/hadjiprocopis/perl-inline-cuda) ) to be installed.
