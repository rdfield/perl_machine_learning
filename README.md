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
Updated from nn_1.pl to use CUDA for Matrix manipulation.  Takes 6 **seconds** per epoch, running on RTX3060 with Ryzen 5 3600 CPU.  95% accuracy.  Run with `perl nn_1_cuda_c.pl`.  Requires the Perl module Inline::CUDA (and all of its prerequistites - see [Inline::CUDA](https://github.com/hadjiprocopis/perl-inline-cuda) ) to be installed.

## nn_1_rocm_c.pl
Some notes about getting ROCm to work on Debian 12:
* uninstall all \*rocm\* packages if installed from a Debian repo (cmake is broken and can't find its config file)
* down load and *apt install* [amdgpu-installer](http://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/amdgpu-install_6.1.60100-1_all.deb) - this is the Ubuntu 20.04 installer
* amdgpu-installer won't work out of the box, and needs a specific version of Python, 3.10.  After trying to install 3.10 via [pyenv](https://github.com/pyenv/pyenv), turns out the static version of Python is needed:
  * https://snapshot.debian.org/archive/debian/20210325T142914Z/pool/main/m/mpdecimal/libmpdec3_2.5.1-2_amd64.deb
  * https://snapshot.debian.org/archive/debian/20230223T205901Z/pool/main/p/python3.10/libpython3.10-minimal_3.10.10-1_amd64.deb
  * https://snapshot.debian.org/archive/debian/20230223T205901Z/pool/main/p/python3.10/libpython3.10-stdlib_3.10.10-1_amd64.deb
  * https://snapshot.debian.org/archive/debian/20230223T205901Z/pool/main/p/python3.10/libpython3.10_3.10.10-1_amd64.deb
* Download and install all of these .deb files manually (apt install <localfilename>)
* Try running amdgpu-installer again
* Fails with DKMS not found
* *dpkg-reconfigure amdgpu-dkms* fixes the problem
* add any non-root user to the video and render groups: usermod -a -G render,video <user>
* reboot

Download the samples from [ROCm examples](https://github.com/ROCm/rocm-examples), and to test cd to the (for example) matrix-multiplication directory and run
* cmake -S . -B build
* cmake --build build
* cmake --install build --prefix install
* cd install/bin
* (in another terminal window run something like: watch -n 1 -c rocm-smi)
* ./hip_matrix_multiplication
* you will see the Power and SCLK values change in rocm-smi to show activity on the GPU.

After ROCm is installed and working, compile the code in the amd_kernel directory (kernels.hip).  Then run perl nn_1_rocm_c.pl.  Takes about 1 minute per epoch running on RX6900XT with Ryzen 5 3600 CPU.
