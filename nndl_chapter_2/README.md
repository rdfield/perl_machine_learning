Updated from the nndl_chapter_1 scripts to run mini batches in a single pass, rather than 1 activation at a time.

CUDA now runs one epoch in 1.1s
ROCM now runs one epoch in 8.5s

Accuracy still ~ 95%

One change in the image loading - using 2D arrays with one column per image in Perl was very memory in efficient, it appears to be better with 1 row per image.  Shuffling is also easier.  It does mean transposing the input batch before processing, but that's done on the GPU so pretty quick.

Installing and pre-requisites as per the chapter 1 README.
