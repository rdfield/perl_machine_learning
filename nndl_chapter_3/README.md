Updated from the nndl_chapter_2 scripts:

* allow choice between Mean Squared Error and Cross Entropy loss functions
* added in evaluation data pass, along with parameters to choose which measures to calculate - cost and accuracy per epoch for testing and validation data
* added weight decay, via the lambda parameter
* save/load weights and biases
* added predict functions for batch and single inputs

With all measures enabled CUDA `perl nn_3_cuda_c.pl` now runs one epoch in 1.38s, or 1.02s with no measures enabled and no validation data.

With all measumres enabled ROCM `perl nn_3_rocm.pl` now runs one epoch in 10s, or 7.6s with no measures and no validation data.

Test performed on Debian 12 with RTX3060 and 6900xt GPUs, using Ryzen 5 3600 CPU.

Accuracy with decay and Cross Entropy enabled now 96%.

Installing and pre-requisites as per the chapter 1 README.

Notes on converting non MNIST images of handwritten digits to a compatible format:

1. image with some handwritten digits - numbers.jpg (copied from https://github.com/arijits148/Handwritten-Digit-Recognition/blob/master/numbers.jpg )
2. used `gthumb` to crop each digit to its own file, resize to 28x28, convert to greyscale and invert the image - producing the number_`<x>`.jpg files
3. ran convert number_`<x>`.jpg number_`<x>`.txt to generate a file with 1 line per pixel showing the RGB value for each pixel
4. ran `perl convert_rgb_to_greyscale.pl number_<x>.txt` to output file called number_`<x>`.dat with 1 byte per pixel (the original image did not have a perfectly white background, so a "filter" was applied in the script where every value below 59 was squashed to 0).

`perl nn_preload_cuda.pl number_<x>.dat`

`perl nn_preload_rocm.pl number_<x>.dat`

shows the prediction using either the CUDA or ROCM generated network for a single image.

`perl nn_preload_cuda_mini_batch.pl`

loads a batch file containing 10 images and outputs the predictions. 
