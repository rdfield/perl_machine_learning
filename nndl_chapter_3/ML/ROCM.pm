use Modern::Perl;
package ML::ROCM;
use Math::Matrix;
use Math::Random;
use Data::Dumper;
use File::Slurp;
use List::Util qw/shuffle/;
use Time::HiRes qw(gettimeofday tv_interval);
use Cwd qw(abs_path);
use JSON;


require Exporter;
our @ISA = qw(Exporter);
our @EXPORT = qw(create_network print_network feedforward backprop update_weights SGD calculate_loss save_network load_network mnist_batch_guess mnist_image_guess);

my $code;

BEGIN {
        $code = <<'EOCODE';
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stddef.h>
#include <algorithm>
#include <iostream>
#include <vector>

#include <cassert>
#include <cstddef>
#include <cstring>

#include "node_typedef.h"
#include "Kernels.h"

int debug = 0;
int loss_function = 1;

int mini_batch_size;

#define HAVE_PERL_VERSION(R, V, S) \
    (PERL_REVISION > (R) || (PERL_REVISION == (R) && (PERL_VERSION > (V) || (PERL_VERSION == (V) && (PERL_SUBVERSION >= (S))))))

float *host_Cost_Derivative;
float *device_Cost_Derivative;
float *host_Cost;
float *device_Cost;
int dirty_dirty_weights = 0;

// This section is boilerplace code to move data from Perl -> C and back again

#define sv_setrv(s, r)  S_sv_setrv(aTHX_ s, r)

static void S_sv_setrv(pTHX_ SV *sv, SV *rv)
{
  sv_setiv(sv, (IV)rv);
#if !HAVE_PERL_VERSION(5, 24, 0)
  SvIOK_off(sv);
#endif
  SvROK_on(sv);
}

int is_array_ref(
        SV *array,
        size_t *array_sz
);
int array_numelts_2D(
        SV *array,
        size_t *_Nd1,
        size_t **_Nd2
);
int array_of_unsigned_int_into_AV(
        size_t *src,
        size_t src_sz,
        SV *dst
);
int array_of_int_into_AV(
        int *src,
        size_t src_sz,
        SV *dst
);

int is_array_ref(
        SV *array,
        size_t *array_sz
){
        if( ! SvROK(array) ){ fprintf(stderr, "is_array_ref() : warning, input '%p' is not a reference.\n", array); return 0; }
        if( SvTYPE(SvRV(array)) != SVt_PVAV ){ fprintf(stderr, "is_array_ref() : warning, input ref '%p' is not an ARRAY reference.\n", array); return 0; }
        // it's an array, cast it to AV to get its len via av_len();
        // yes, av_len needs to be bumped up
        int asz = 1+av_len((AV *)SvRV(array));
        if( asz < 0 ){ fprintf(stderr, "is_array_ref() : error, input array ref '%p' has negative size!\n", array); return 0; }
        *array_sz = (size_t )asz;
        return 1; // success, it is an array and size returned by ref, above
}

#define array_numelts_1D(A,B) (!is_array_ref(A,B))

int array_numelts_2D(
        SV *array,
        size_t *_Nd1,
        size_t **_Nd2
){
        size_t anN, anN2, *Nd2 = NULL;

        if( ! is_array_ref(array, &anN) ){
           fprintf(stderr, "is_array_ref_2D() : error, call to is_array_ref() has failed for array '%p'.\n", array);
           return 1;
        }

        if( *_Nd2 == NULL ){
           if( (Nd2=(size_t *)malloc(anN*sizeof(size_t))) == NULL ){
               fprintf(stderr, "array_numelts_2D() : error, failed to allocate %zu bytes for %zu items for Nd2.\n", anN*sizeof(size_t), anN);
               return 1;
           }
        } else Nd2 = *_Nd2;
        AV *anAV = (AV *)SvRV(array);
        size_t *pNd2 = &(Nd2[0]);
        for(size_t i=0;i<anN;i++,pNd2++){
           SV *subarray = *av_fetch(anAV, i, FALSE);
           if( ! is_array_ref(subarray, &anN2) ){
              fprintf(stderr, "is_array_ref_2D() : error, call to is_array_ref() has failed for [%p][%p], item %zu.\n", array, subarray, i);
              if(*_Nd2==NULL) free(Nd2);
              return 1;
           }
           *pNd2 = anN2;
        }
        if( *_Nd2 == NULL ) *_Nd2 = Nd2;
        *_Nd1 = anN;
        return 0; // success
}

int array_of_int_into_AV(
        int *src,
        size_t src_sz,
        SV *dst
){
        size_t dst_sz;
        if( ! is_array_ref(dst, &dst_sz) ){ fprintf(stderr, "array_of_int_into_AV() : error, call to is_array_ref() has failed.\n"); return 1; }
        AV *dstAV = (AV *)SvRV(dst);
        for(size_t i=0;i<src_sz;i++){
                av_push(dstAV, newSViv(src[i]));
        }
        return 0; // success
}
// end of Perl -> C -> Perl section
void print_2D_array(float *foo, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%+.5f\t", foo[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}


node_t * head = NULL;
node_t * tail = NULL;

// these are where the initial input for each "feedforward" will be stored
float *host_x; 
float *device_x; 
float *host_x_transposed; 
float *device_x_transposed; 
SV    *perl_x; 
// these are where the target for each "feedforward" will be stored
float *host_y; 
float *device_y; 
float *host_y_transposed; 
float *device_y_transposed; 
SV    *perl_y; 

node* create_node(int insize, int outsize, SV *biases, SV *weights, int batch_size)
{
    AV *av;
    float *pd;
    size_t i,j;
    size_t AH, AW, *AWs = NULL;
    SV *subav, *subsubav;

    node_t * new_node = (node_t *)malloc(sizeof(node_t));
    new_node->input_size = insize;
    new_node->output_size = outsize;
    new_node->perl_Bias = biases;
    new_node->perl_Weights = weights;
// convert Perl bias array to C array of floats and push it onto the GPU
    new_node->device_Bias = gpu_device_malloc(sizeof(float)*outsize);
    new_node->host_Bias = gpu_host_malloc(sizeof(float)*outsize*1);

    array_numelts_2D(new_node->perl_Bias, &AH, &AWs);
    AW = AWs[0];

    pd = &(new_node->host_Bias[0]);
    av = (AV *)SvRV(new_node->perl_Bias);
    for(i=0;i<AH;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<AW;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
// convert Perl weight array to C array of floats and push it onto the GPU
    new_node->host_Weights = gpu_host_malloc(sizeof(float)*outsize*insize);
    new_node->device_Weights = gpu_device_malloc(sizeof(float)*insize*outsize);
    pd = &(new_node->host_Weights[0]);
    av = (AV *)SvRV(new_node->perl_Weights);
    for(i=0;i<outsize;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<insize;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
    gpu_memcpy_to_device(new_node->host_Bias, new_node->device_Bias, outsize*sizeof(float));
    gpu_memcpy_to_device(new_node->host_Weights, new_node->device_Weights, insize*outsize*sizeof(float));
// reserve memory for output and activated output (both 1 x outsize)
    new_node->host_Output = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Output = gpu_device_malloc(sizeof(float)*batch_size*outsize);
    new_node->host_Activated_Output = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Activated_Output = gpu_device_malloc(sizeof(float)*batch_size*outsize);
    new_node->host_Activated_Output_Derivative = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Activated_Output_Derivative = gpu_device_malloc(sizeof(float)*batch_size*outsize);
    new_node->host_Activated_Output_Transposed = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Activated_Output_Transposed = gpu_device_malloc(sizeof(float)*batch_size*outsize);
// reserve memory for deriviatives, bias = 1 * outsize, weight = insize * outsize
    new_node->host_Weights_Derivative = gpu_host_malloc(sizeof(float)*outsize*insize);
    new_node->device_Weights_Derivative = gpu_device_malloc(sizeof(float)*insize*outsize);
    new_node->host_Bias_Derivative = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Bias_Derivative = gpu_device_malloc(sizeof(float)*batch_size*outsize);
// reserve memory for transposed Weights
    new_node->host_Weights_Transposed = gpu_host_malloc(sizeof(float)*outsize*insize);
    new_node->device_Weights_Transposed = gpu_device_malloc(sizeof(float)*insize*outsize);
// reserve memory for temporary derivative calculation
    new_node->host_Delta = gpu_host_malloc(sizeof(float)*outsize*batch_size);
    new_node->device_Delta = gpu_device_malloc(sizeof(float)*batch_size*outsize);
         
    new_node->next = NULL; 
    new_node->prev = NULL; 
    return new_node;
}        

int add_node(int insize, int outsize, SV *biases, SV *weights, int batch_size)
{

    node_t* new_node = create_node(insize, outsize, biases, weights, batch_size); 
    if (new_node == NULL) {
       std::cerr << "no node created for " << insize << " x " << outsize << std::endl;
       return 0;
    }
    if (tail == NULL) { 
        head = new_node; 
        tail = new_node; 
    } 
    else { 
        new_node->prev = tail; 
        tail->next = new_node; 
        tail = new_node; 
    } 
    return 1;
}

void reset_derivatives() {
    node_t * current = head;
    while (current != NULL) {
       for (int i = 0; i < current->output_size; ++i) {
           for (int j = 0; j < current->input_size; ++j) {
              current->host_Weights_Derivative[i *  current->input_size + j] = 0;
           }
           current->host_Bias_Derivative[i] = 0;
        }
        gpu_memcpy_to_device(current->host_Bias_Derivative, current->device_Bias_Derivative, current->output_size*sizeof(float));
        gpu_memcpy_to_device(current->host_Weights_Derivative, current->device_Weights_Derivative, current->input_size * current->output_size * sizeof(float));
        current = current->next;
    }
}

int reserve_input_memory(int insize, int outsize, int batch_size)
{  
// memory for input array

    host_x = gpu_host_malloc(sizeof(float)*insize*batch_size); 
    host_x_transposed = gpu_host_malloc(sizeof(float)*insize*batch_size);
    device_x = gpu_device_malloc(sizeof(float)*insize*batch_size);
    device_x_transposed = gpu_device_malloc(sizeof(float)*insize*batch_size);

    host_y = gpu_host_malloc(sizeof(float)*outsize*batch_size); 
    host_y_transposed = gpu_host_malloc(sizeof(float)*outsize*batch_size); 
    device_y = gpu_device_malloc(sizeof(float)*outsize*batch_size);
    device_y_transposed = gpu_device_malloc(sizeof(float)*outsize*batch_size);

    host_Cost_Derivative = gpu_host_malloc(sizeof(float)*outsize*batch_size); 
    device_Cost_Derivative = gpu_device_malloc(sizeof(float)*outsize*batch_size);

    host_Cost = gpu_host_malloc(sizeof(float)*outsize*batch_size); 
    device_Cost = gpu_device_malloc(sizeof(float)*outsize*batch_size);

    mini_batch_size = batch_size;
    return 1;
}

int load_input(SV *x, int elements) 
{
// insize x 1 input array
    AV *av;
    float *pd;
    size_t i,j,insize;
    SV *subav, *subsubav; 
    mini_batch_size = elements;
    pd = &(host_x_transposed[0]);
    av = (AV *)SvRV(x);
    insize = head->input_size;

    for(i=0;i<elements;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<insize;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
    // now transfer to device
    gpu_memcpy_to_device(host_x_transposed, device_x_transposed, mini_batch_size*insize*sizeof(float));
    run_gpu_transpose_2D_array(device_x_transposed, device_x, mini_batch_size, insize);
    if (debug == 1) {
       gpu_memcpy_from_device(host_x, device_x, insize*mini_batch_size*sizeof(float));
       std::cout << "host_x"<<std::endl;
       print_2D_array(host_x, insize, mini_batch_size);
    }

    return 1;
}

int load_target(SV *y)
{
// outsize x 1 input array
    AV *av;
    float *pd; 
    size_t i,j,outsize;
    SV *subav, *subsubav; 
    pd = &(host_y_transposed[0]);
    av = (AV *)SvRV(y);
    outsize = tail->output_size;
    for(i=0;i<mini_batch_size;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<outsize;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
    // now transfer to device
    gpu_memcpy_to_device(host_y_transposed, device_y_transposed, mini_batch_size*outsize*sizeof(float));
    run_gpu_transpose_2D_array(device_y_transposed, device_y, mini_batch_size, outsize);
    if (debug == 1) {
printf("host_y %p device_y %p\n", host_y, device_y);
       gpu_memcpy_from_device(host_y, device_y, outsize*mini_batch_size*sizeof(float));
       std::cout << "host_y"<<std::endl;
       print_2D_array(host_y, outsize, mini_batch_size);
    }

    return 1;
}  


void run_feed_forward() {
    node_t * current = head;
    
    float *activation;
    activation = device_x;
    run_gpu_transpose_2D_array(activation, device_x_transposed, head->input_size,1);
    while (current != NULL) {
        run_gpu_linear( activation, current->device_Weights, current->device_Bias, current->device_Output, current->output_size, current->input_size, mini_batch_size );
        if (debug == 1) {
           if (current == head) {
              gpu_memcpy_from_device(host_x, device_x, head->input_size*mini_batch_size*sizeof(float));
              std::cout << "Initial Input Activations" << std::endl;
              print_2D_array(host_x, head->input_size, mini_batch_size);
           } else {
              gpu_memcpy_from_device(current->prev->host_Activated_Output, current->prev->device_Activated_Output, current->input_size*mini_batch_size*sizeof(float));
              std::cout << "Input Activations" << std::endl;
              print_2D_array(current->prev->host_Activated_Output, current->input_size, mini_batch_size);
           }
           gpu_memcpy_from_device(tail->host_Bias, tail->device_Bias, tail->output_size*sizeof(float));
           std::cout << "Bias" << std::endl;
           print_2D_array(current->host_Bias, current->output_size, 1);
           gpu_memcpy_from_device(tail->host_Weights, tail->device_Weights, tail->input_size*tail->output_size*sizeof(float));
           std::cout << "Weights" << std::endl;
           print_2D_array(current->host_Weights, current->output_size, current->input_size);
           gpu_memcpy_from_device(current->host_Output, current->device_Output, mini_batch_size * current->output_size * sizeof(float));
           std::cout << "Output before activation" << std::endl;
           print_2D_array(current->host_Output, current->output_size, mini_batch_size);
        }

        gpu_sigmoid( current->device_Output, current->device_Activated_Output, current->output_size, mini_batch_size );
        if (debug == 1) {
           gpu_memcpy_from_device(current->host_Activated_Output, current->device_Activated_Output, mini_batch_size * current->output_size * sizeof(float));
           std::cout << "Output after activation" << std::endl;
           print_2D_array(current->host_Activated_Output,  current->output_size, mini_batch_size );
        }

        activation = current->device_Activated_Output;
        current = current->next;
    }
}

void run_backpropagation() {
   node_t *current = tail;

   //float *delta = device_Cost_Derivative;
   gpu_memcpy_intra_device( device_Cost_Derivative, current->device_Delta, mini_batch_size * current->output_size*sizeof(float));
   gpu_memcpy_intra_device( device_Cost_Derivative, current->device_Bias_Derivative, mini_batch_size * current->output_size*sizeof(float));
   if (debug == 1) {
      gpu_memcpy_from_device(current->prev->host_Activated_Output, current->prev->device_Activated_Output, tail->input_size*mini_batch_size*sizeof(float));
      std::cout << "Initial Backpass Input Activations " << std::endl;
      print_2D_array(current->prev->host_Activated_Output,  tail->input_size, mini_batch_size);
   }
   run_gpu_transpose_2D_array(current->prev->device_Activated_Output, current->prev->device_Activated_Output_Transposed, current->prev->output_size,mini_batch_size);
   if (debug == 1) {
      gpu_memcpy_from_device(current->prev->host_Activated_Output_Transposed, current->prev->device_Activated_Output_Transposed, tail->input_size*mini_batch_size*sizeof(float));
      std::cout << "Initial Backpass Input Activations Transposed" << std::endl;
      print_2D_array(current->prev->host_Activated_Output_Transposed, mini_batch_size, tail->input_size ); // it's transposed so now outsize x rows, rather than rows x outsize
      printf("host_Delta %p\n", current->host_Delta);
      printf("device_Delta %p\n", current->device_Delta);
      printf("outsize %d\n", tail->output_size);
      gpu_memcpy_from_device(current->host_Delta, current->device_Delta, tail->output_size*mini_batch_size*sizeof(float));
      std::cout << "Initial Backpass Input Delta" << std::endl;
      print_2D_array(current->host_Delta, tail->output_size, mini_batch_size);
   }

   run_gpu_weight_derivative( current->device_Delta, current->prev->device_Activated_Output_Transposed, current->device_Weights_Derivative, current->output_size,  mini_batch_size,current->input_size ); 
   if (debug == 1) {
      gpu_memcpy_from_device(current->host_Weights_Derivative, current->device_Weights_Derivative, current->input_size*current->output_size*sizeof(float));
      std::cout << "Last Layer Backpass Weights Derivative" << std::endl;
      print_2D_array(current->host_Weights_Derivative, current->output_size, current->input_size);
   }

   current = current->prev;
   while (current != NULL) {
// do the back prop
      run_gpu_sigmoid_prime(current->device_Activated_Output, current->device_Activated_Output_Derivative,current->output_size, mini_batch_size);
      if (debug == 1) {
         printf("host_Activated_Output_Derivative %p\n", current->host_Activated_Output_Derivative);
         printf("device_Activated_Output_Derivative %p\n", current->device_Activated_Output_Derivative);
         printf("outsize %d\n", current->output_size);
         gpu_memcpy_from_device(current->host_Activated_Output_Derivative, current->device_Activated_Output_Derivative, mini_batch_size*current->output_size*sizeof(float));
         std::cout << "Activated Output Derivative" << std::endl;
         print_2D_array(current->host_Activated_Output_Derivative, current->output_size, mini_batch_size);
      }

      float *sp = current->device_Activated_Output_Derivative;
      run_gpu_transpose_2D_array(current->next->device_Weights, current->next->device_Weights_Transposed, current->next->output_size, current->next->input_size);
      if (debug == 1) {
         gpu_memcpy_from_device(current->next->host_Weights_Transposed, current->next->device_Weights_Transposed, current->next->input_size*current->next->output_size*sizeof(float));
         std::cout << "Weights (next) transposed" << std::endl;
         print_2D_array(current->next->host_Weights_Transposed, current->next->input_size, current->next->output_size);
      }
      run_gpu_derivative(current->next->device_Weights_Transposed, current->next->device_Delta, sp, current->device_Delta, current->output_size, current->next->output_size,mini_batch_size); // wT x delta * sp (e.g. 30x10 x 10x1 => 30 x 1 * 30 x 1)
      if (debug == 1) {
         gpu_memcpy_from_device(current->host_Delta, current->device_Delta, mini_batch_size*current->output_size*sizeof(float));
         std::cout << "New delta" << std::endl;
         print_2D_array(current->host_Delta, current->output_size, mini_batch_size);
      }

      float *activation;
      if (current->prev != NULL) {
         run_gpu_transpose_2D_array(current->prev->device_Activated_Output, current->prev->device_Activated_Output_Transposed, current->input_size,mini_batch_size);
         activation = current->prev->device_Activated_Output_Transposed;
      } else {
         activation = device_x_transposed;
      }
      gpu_memcpy_intra_device( current->device_Delta, current->device_Bias_Derivative, mini_batch_size * current->output_size*sizeof(float));
      run_gpu_matmul(current->device_Delta, activation, current->device_Weights_Derivative, current->output_size, mini_batch_size, current->input_size);
      if (debug == 1) {
         gpu_memcpy_from_device(current->host_Weights_Derivative, current->device_Weights_Derivative, current->output_size*current->input_size*sizeof(float));
         std::cout << "New Weights Derivative" << std::endl;
         print_2D_array(current->host_Weights_Derivative, current->output_size, current->input_size);
      }

      current = current->prev;
   }
}

void run_update_weights_and_biases(float modifier, float decay) {
   node_t * current = head;

   while (current != NULL) {
      run_gpu_update_weights(modifier, decay, current->device_Weights, current->device_Weights_Derivative, current->output_size, current->input_size);
      run_gpu_update_biases(modifier, current->device_Bias, current->device_Bias_Derivative, current->output_size, mini_batch_size);
      current = current->next;
   }
   dirty_dirty_weights = 1;
}

int get_last_activated_output( SV *R ) {
  
    // Transfer results from host to perl
    AV *av, *av2;
    float *pd;
    size_t i,j,RH,RW, asz;
    
    RW = mini_batch_size;
    RH = tail->output_size;
   
// copy device data back to the host before loading the Perl values

    gpu_memcpy_from_device(tail->host_Activated_Output, tail->device_Activated_Output, RW*RH*sizeof(float));
   
    if( is_array_ref(R, &asz) ){
            av = (AV *)SvRV(R);
            if( asz > 0 ){
               av_clear(av);
            }
    } else if( SvROK(R) ){
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(SvRV(R), (SV *)av);
    } else {
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(R, (SV *)av);
    }
    
    pd = &(tail->host_Activated_Output[0]);
    av = (AV *)SvRV(R);
    for(i=0;i<RH;i++){ // for each row
        av2 = newAV(); // make a new array for each row
        av_extend(av2, RH); // extend it to hold #cols items (RW)
        // LeoNerd's suggestion
        av_push(av, newRV_noinc((SV *)av2)); // insert it into the top Array
        for(j=0;j<RW;j++){ // for the cols of that row
            av_store(av2, j, newSVnv(*pd));
            pd++;
        }
    }
    if (debug == 1) {
       gpu_memcpy_from_device(tail->prev->host_Activated_Output, tail->prev->device_Activated_Output, tail->input_size*mini_batch_size*sizeof(float));
       std::cout << "Input Activations" << std::endl;
       print_2D_array(tail->host_Activated_Output, mini_batch_size, tail->output_size);
       gpu_memcpy_from_device(tail->host_Bias, tail->device_Bias, tail->output_size*sizeof(float));
       std::cout << "Bias" << std::endl;
       print_2D_array(tail->host_Bias, 1, tail->output_size);
       gpu_memcpy_from_device(tail->host_Weights, tail->device_Weights, tail->input_size*tail->output_size*sizeof(float));
       std::cout << "Weights" << std::endl;
       print_2D_array(tail->host_Weights, tail->output_size, tail->input_size);
       gpu_memcpy_from_device(tail->host_Output, tail->device_Output, RW*RH*sizeof(float));
       std::cout << "Output before activation" << std::endl;
       print_2D_array(tail->host_Output, RH, RW);
       std::cout << "Activtated Output" << std::endl;
       print_2D_array(tail->host_Activated_Output, RH, RW);
    }

    return 0;
}

float calculate_cost(){
      
   gpu_calculate_cost(tail->device_Activated_Output, device_y, device_Cost, tail->output_size, mini_batch_size, loss_function);

   if (debug == 1) { 
       gpu_memcpy_from_device(tail->host_Activated_Output, tail->device_Activated_Output, mini_batch_size * tail->output_size * sizeof(float) );
       std::cout << "final layer activation" << std::endl;
       print_2D_array(tail->host_Activated_Output, tail->output_size, mini_batch_size);
       std::cout << "targets" << std::endl;
       print_2D_array(host_y, tail->output_size, mini_batch_size);
   }
   gpu_memcpy_from_device(host_Cost, device_Cost, mini_batch_size * tail->output_size * sizeof(float) );
   float sum = 0;
   for (int i = 0; i < mini_batch_size; i++) {
      for (int j = 0; j < tail->output_size; j++) {
         sum += host_Cost[i * tail->output_size + j];
      }
   }
   if (debug == 1) {
       std::cout << "cost calc" << std::endl;
       print_2D_array(host_Cost, mini_batch_size, tail->output_size);
       std::cout << "sum of cost before weights calc " << sum << std::endl;
   }
   return sum;
}  

float calculate_weights_cost() {
   node_t * current = head;
   float sum = 0;
   while (current != NULL) {
      if (dirty_dirty_weights == 1) {
         gpu_memcpy_from_device(current->host_Weights, current->device_Weights, current->input_size * current->output_size*sizeof(float));
      }
      for (int i = 0; i < current->input_size; i++) {
         for (int j = 0; j < current->output_size; j++) {
            sum += powf(current->host_Weights[i * current->output_size + j],2);
         }
      }
      current = current->next;
   }
   dirty_dirty_weights = 0;
   return sum;
}     

void calculate_cost_derivative() {
   gpu_calculate_cost_and_derivative(tail->device_Activated_Output, device_y, device_Cost_Derivative, tail->output_size, mini_batch_size, loss_function);
   if (debug == 1) {
       gpu_memcpy_from_device(tail->host_Activated_Output, tail->device_Activated_Output, mini_batch_size * tail->output_size * sizeof(float));
       std::cout << "final layer activation" << std::endl;                      
       print_2D_array(tail->host_Activated_Output, tail->output_size, mini_batch_size);
       gpu_memcpy_from_device(host_Cost_Derivative, device_Cost_Derivative, mini_batch_size * tail->output_size * sizeof(float));
       std::cout << "cost calc" << std::endl;
       print_2D_array(host_Cost_Derivative, mini_batch_size, tail->output_size);
   }
}

void set_debug_on() {
   debug = 1;
}

void set_debug_off() {
   debug = 0;
}

void set_loss(int funcno) { 
   loss_function = funcno;
}

void get_weights(SV *R, int i) {
   node_t * current = head;
   AV *av, *av2;
   float *pd;
   size_t j,RH,RW, asz;

   for (int skip = 0;skip < i; skip++) {
      current = current->next;
   }
   if( is_array_ref(R, &asz) ){
            av = (AV *)SvRV(R);
            if( asz > 0 ){
               av_clear(av);
            }
   } else if( SvROK(R) ){
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(SvRV(R), (SV *)av);
   } else {
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(R, (SV *)av);
   }

   RH = current->output_size;
   RW = current->input_size;
   gpu_memcpy_from_device(current->host_Weights, current->device_Weights, current->output_size*current->input_size*sizeof(float));

   pd = &(current->host_Weights[0]);
   av = (AV *)SvRV(R);
   for(i=0;i<RH;i++){ // for each row
       av2 = newAV(); // make a new array for each row
       av_extend(av2, RH); // extend it to hold #cols items (RW)
       // LeoNerd's suggestion
       av_push(av, newRV_noinc((SV *)av2)); // insert it into the top Array
       for(j=0;j<RW;j++){ // for the cols of that row
           av_store(av2, j, newSVnv(*pd));
           pd++;
       }
   }
}

void get_biases(SV *R, int i) {
   node_t * current = head;
   AV *av, *av2;
   float *pd;
   size_t j,RH,RW, asz;

   for (int skip = 0;skip < i; skip++) {
      current = current->next;
   }
   if( is_array_ref(R, &asz) ){
            av = (AV *)SvRV(R);
            if( asz > 0 ){
               av_clear(av);
            }
   } else if( SvROK(R) ){
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(SvRV(R), (SV *)av);
   } else {
            av = newAV();
            // LeoNerd's suggestion:
            sv_setrv(R, (SV *)av);
   }

   RH = current->output_size;
   RW = 1;
   gpu_memcpy_from_device(current->host_Bias, current->device_Bias, current->output_size*sizeof(float));

   pd = &(current->host_Bias[0]);
   av = (AV *)SvRV(R);
   for(i=0;i<RH;i++){ // for each row
       av2 = newAV(); // make a new array for each row
       av_extend(av2, RH); // extend it to hold #cols items (RW)
       // LeoNerd's suggestion
       av_push(av, newRV_noinc((SV *)av2)); // insert it into the top Array
       for(j=0;j<RW;j++){ // for the cols of that row
           av_store(av2, j, newSVnv(*pd));
           pd++;
       }
   }
}

EOCODE
}

use Inline CPP => Config =>
        BUILD_NOISY => 1,
        force_build => 1,
        clean_after_build => 0,
        warnings => 10,
        INC => "-I" . abs_path("./inc") . " -I" . abs_path("./amd_kernel"), 
        LIBS => "-L" . abs_path("./amd_kernel") . " -lKernels"
;

use Inline CPP => $code;

my %loss_functions = (
                       "quadratic" => 1,
                       "cel" => 2
                     );

my %network_init_params;

sub create_network {
   my ($sizes, @options) = @_;
   $network_init_params{ sizes } = $sizes;
   my %params = @options;
   $params{batch_size} ||= 10;
   my $scale_factor = 1;
   if (defined($params{weight_init}) and $params{weight_init} eq "scaled") {
      $scale_factor = sqrt($params{batch_size});
   }
   foreach my $i (0 .. ($#{$sizes} - 1)) {
      # "input size" = $sizes->[$i]
      # "output size" = $sizes->[$i + 1]
      if (!defined($params{ bias }->[$i])) {
         # this will be an "output size" x 1 array
         $params{bias}->[$i] = Math::Matrix->new([random_normal($sizes->[$i + 1])])->transpose()->as_array();
      }
      if (!defined($params{ weights }->[$i])) {
         # this will be an "output size" x "input size" array
         my $iw = [];
         foreach my $s (1 .. $sizes->[$i + 1]) {
            push @$iw, [map { $_ / $scale_factor } random_normal( $sizes->[$i] ) ];
         }
         $params{weights}->[$i] = $iw;
      }
      return 0 unless add_node($sizes->[$i], $sizes->[$i + 1], $params{bias}->[$i], $params{weights}->[$i], $params{batch_size});
   }
   # reserve RAM for initial input
   reset_derivatives();
   if ($params{debug} and $params{debug} == 1) {
      set_debug_on();
   } else {
      set_debug_off();
   }
   if ($params{loss} =~ /(CrossEntropy|CrossEntropyLoss|CEL)/i) {
      set_loss($loss_functions{cel});
      $network_init_params{loss} = "cel";
   } else {
      set_loss($loss_functions{quadratic});
      $network_init_params{loss} = "mse";
   }
   return 0 unless reserve_input_memory($sizes->[0], $sizes->[-1], $params{batch_size});
   return 1;
}

sub print_network {
   print_list();
}

sub feedforward {
   my $xy = shift;
   # convert input to C, put into already reserved memory
   my (@x, @y);
   my $elements = 0;
   foreach my $input (@$xy) {
      push @x, $input->[0];
      push @y, $input->[1];
      $elements++;
   }
   return unless load_input(\@x, $elements);
   return unless load_target(\@y);
   # run the forward pass of the network
   run_feed_forward();
   #my $last_activated_output = [];
   #get_last_activated_output($last_activated_output) ;
   #Math::Matrix->new($last_activated_output)->print("Last activated output");
}

sub validation_feedforward {
   my $xy = shift;
   # convert input to C, put into already reserved memory
   my (@x, @y);
   my $elements = 0;
   foreach my $input (@$xy) {
      push @x, $input->[0];
      push @y, $input->[1];
      $elements++;
   }
say "loading input";
   return unless load_input(\@x, $elements);
say "loading target";
   return unless load_target(\@y);
   # run the forward pass of the network
say "running feed_forward";
   run_feed_forward();
   my $last_activated_output = [];
   get_last_activated_output($last_activated_output) ;
   return $last_activated_output;
}

sub calculate_loss {
   calculate_cost_derivative();
}

sub backprop {
   run_backpropagation();
}

sub update_weights {
  # run at end of mini batch
  # don't forget to zero the derivatives!
  my %params = @_;
  $params{batch_size} ||= 10;
  $params{learning_rate} ||= 3;
  $params{decay} ||= 1;
  run_update_weights_and_biases( $params{learning_rate} / $params{batch_size}, $params{decay} );
  reset_derivatives();
}

sub update_mini_batch {
   my ($mb, $eta, $j, $ctr, $decay) = @_;
   feedforward($mb);
   calculate_loss();
   backprop();
   update_weights( batch_size => scalar(@$mb), learning_rate => $eta, decay => $decay );
}

sub argmax {
   my $arr = shift;
   my @max;
   foreach my $i (0 .. $#{$arr->[0]}) {
      my $max = $arr->[0][$i];
      $max[$i] = 0;
      my $idx = 0;
      foreach my $j ( 0 .. $#$arr) {
         if ($arr->[$j][$i] > $max) {
            $max = $arr->[$j][$i];
            $max[$i] = $j;
         }
      }
   }
   return \@max;
}

my @evaluation_batches;
my @evaluation_targets;
my @testing_batches;
my @testing_targets;


sub total_cost {
   my $data = shift;
   my $targets = shift;
   my $lambda = shift;
   my $cost = shift;
   my $accuracy = shift;
   my $data_len = 0;
   my $total_cost = 0;
   my $successes = 0;
   foreach my $i (0 .. $#$data) {
      my $calc;
      if ($accuracy) {
         $calc = validation_feedforward($data->[$i]);
      } else {
         feedforward($data->[$i]);
      }
      if ($cost) {
         $total_cost += calculate_cost();
         $data_len += scalar(@{$data->[$i]});
      }
      if ($accuracy) {
         my $max_calc_idx = argmax($calc);
         my $max_target_idx = argmax($targets->[$i]);
         foreach my $i (0 .. $#$max_calc_idx) {
            $successes++ if  $max_calc_idx->[$i] == $max_target_idx->[$i];
         }
      }
   }
   if ($cost) {
      $total_cost += calculate_weights_cost();
      $total_cost /= $data_len;
   }
   return $total_cost, $successes;
}

sub _cache_eval_data {
   my $mini_batch_size = shift;
   my $data = shift;
   my $data_cache = shift;
   my $target_cache = shift;
   my $k = 0;
   my $data_size = scalar(@$data);
   while (($k * $mini_batch_size) < $data_size) {
      push @$data_cache, [@$data[($k * $mini_batch_size) .. ((($k + 1) * $mini_batch_size) - 1)]];
      my @y;
      foreach my $i (0 .. ($mini_batch_size - 1)) {
         push @y, $data_cache->[-1][$i][1];
      }
      push @$target_cache, Math::Matrix->new(\@y)->transpose->as_array();
      $k++;
   }
}

sub SGD {
   my $training_data = shift;
   my $epochs = shift;
   my $mini_batch_size = shift;
   my $eta = shift;
   my %params = @_;
   $params{load_from_file} ||= 0;
   my $test_data = $params{test_data};
   my $n = scalar(@$training_data);
   my $n_test;
   if ($test_data) {
      $n_test = scalar(@$test_data);
      _cache_eval_data($mini_batch_size, $test_data, \@testing_batches, \@testing_targets);
   }
   my $evaluation_data = $params{evaluation_data};
   my $n_eval;
   if ($evaluation_data) {
      $n_eval = scalar(@$evaluation_data);
      _cache_eval_data($mini_batch_size, $evaluation_data, \@evaluation_batches, \@evaluation_targets);
   }
   $params{lambda} ||= 0;
   my $decay = 1 - $eta * ( $params{lambda} / $n_test );
   my (@evaluation_cost, @evaluation_accuracy, @training_cost, @training_accuracy);
   foreach my $j (1 .. $epochs) {
      my $start = [ gettimeofday ];
      my @training_data = shuffle(@$training_data);
      my @mini_batches;  
      my $k = 0;
      while (scalar(@training_data)) {
         push @mini_batches, [ splice @training_data, 0 , $mini_batch_size ];
      } 
      my $ctr = 1;
      foreach my $mb (@mini_batches) {
         update_mini_batch($mb, $eta, $j, $ctr, $decay);
         $ctr++;
      } 
      say "Epoch $j training complete";
      if (defined($params{monitor_training_cost}) or defined($params{monitor_training_accuracy})) {
         my ($cost,$accuracy) = total_cost( \@testing_batches, \@testing_targets, $params{ lambda }, $params{monitor_training_cost}, $params{monitor_training_accuracy} );
         push @training_cost, $cost;
         say "Cost on training data $cost" if defined($params{monitor_training_cost});
         say "Accuracy on training data $accuracy / $n_test" if defined($params{monitor_training_accuracy})
      }
      if (defined($params{monitor_evaluation_cost}) or defined($params{monitory_evaluation_accuracy})) {
         my ($cost,$accuracy) = total_cost( \@evaluation_batches, \@evaluation_targets, $params{ lambda },  $params{monitor_evaluation_cost}, $params{monitor_evaluation_accuracy} );
         push @evaluation_cost, $cost;
         say "Cost on evaluation data $cost" if defined($params{monitor_evaluation_cost});
         say "Accuracy on evaluation data $accuracy / $n_eval" if defined($params{monitor_evaluation_accuracy})
      }
      say "epoch time = " . tv_interval($start , [gettimeofday]);
   }
   return { evaluation_cost => \@evaluation_cost,
            evaluation_accuracy => \@evaluation_accuracy,
            training_cost => \@training_cost,
            training_accuracy => \@training_accuracy };

}

sub save_network {
   my $filename = shift;
   my $data = { loss => $network_init_params{loss}, sizes => $network_init_params{sizes} };
   foreach my $i (0 .. ($#{$network_init_params{sizes}} - 1)) {
      my $W = [];
      get_weights($W, $i);
      push @{$data->{weights}}, $W;
      my $B = [];
      get_biases($B, $i);
      push @{$data->{bias}}, $B;
   }
   open(FILE, ">", $filename);
   print FILE to_json($data);
   close FILE;
}   
    
sub load_network {
   my $filename = shift;
   my $data = from_json(scalar(read_file($filename)));
   create_network(delete $data->{sizes}, %$data);
}   
    
sub mnist_batch_guess {
# use if the data supplied is in the same format as the mnist batches
# otherwise use mnist_image_guess
   my $data = shift;
   my $calc = validation_feedforward($data);
   return argmax($calc);
}   
    
sub mnist_image_guess {
   my $data = shift;
   # expecting an array of 784 pixel values scaled to the 0-1 range
set_debug_on();
   my @batch;
   $batch[0]->[0] = $data;
   $batch[0]->[1] = [(0) x 10]; # validation expects to see a target array, but it isn't needed, so just make it zeros.
say "calling validation_feedforward";
   my $calc = validation_feedforward(\@batch);
say "calc = " . Dumper($calc);
   return argmax($calc);
}

1;
