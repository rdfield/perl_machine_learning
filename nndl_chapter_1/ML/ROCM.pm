use Modern::Perl;
package ML::ROCM;
use Math::Matrix;
use Math::Random;
use Data::Dumper;
use List::Util qw/shuffle/;
use Time::HiRes qw(gettimeofday tv_interval);
use Cwd qw(abs_path);


require Exporter;
our @ISA = qw(Exporter);
our @EXPORT = qw(create_network print_network feedforward backprop show_final_weights update_weights SGD evaluate);

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

#define HAVE_PERL_VERSION(R, V, S) \
    (PERL_REVISION > (R) || (PERL_REVISION == (R) && (PERL_VERSION > (V) || (PERL_VERSION == (V) && (PERL_SUBVERSION >= (S))))))

float *host_Cost_Derivative;
float *device_Cost_Derivative;

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

node_t * head = NULL;
node_t * tail = NULL;

// these are where the initial input for each "feedforward" will be stored
float *host_x; 
float *device_x; 
float *device_x_transposed; 
SV    *perl_x; 
// these are where the target for each "feedforward" will be stored
float *host_y; 
float *device_y; 
SV    *perl_y; 

int add_node(int insize, int outsize, SV *biases, SV *weights)
{
// convert the input Perl arrays to C arrays
    AV *av;
    float *pd;
    size_t i,j;
    size_t AH, AW, *AWs = NULL;
    SV *subav, *subsubav;

    node_t *new_node;
    float *host_Bias;
    host_Bias = (float *)malloc(sizeof(float) * outsize); 
    float *host_Weights;
    host_Weights = (float *)malloc(sizeof(float) * insize * outsize); 

    array_numelts_2D(biases, &AH, &AWs);
    AW = AWs[0];

    pd = &(host_Bias[0]);
    av = (AV *)SvRV(biases);
    for(i=0;i<AH;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<AW;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }

    pd = &(host_Weights[0]);
    av = (AV *)SvRV(weights);
    for(i=0;i<outsize;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<insize;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }


    new_node = gpu_create_node(insize, outsize, host_Bias, host_Weights); 

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

int reserve_input_memory(int insize, int outsize)
{  
// memory for input array

std::cout << "reserving " << sizeof(float)*insize << " for host_x " << std::endl;
    // gpu_host_malloc(host_x,sizeof(float)*insize); 
    host_x = (float *)malloc(sizeof(float) * insize); 

    host_y = gpu_host_malloc(sizeof(float)*outsize); 
    host_Cost_Derivative = gpu_host_malloc(sizeof(float)*outsize); 
std::cout << "reserving " << sizeof(float)*insize << " for device_x " << std::endl;
    device_x = gpu_device_malloc(sizeof(float)*insize);
printf("device_x, reserve_input_memory = %p\n",device_x);
    device_x_transposed = gpu_device_malloc(sizeof(float)*insize);
    device_y = gpu_device_malloc(sizeof(float)*outsize);
    device_Cost_Derivative = gpu_device_malloc(sizeof(float)*outsize);
    return 1;
}

int load_input(SV *x) 
{
// insize x 1 input array
    AV *av;
    float *pd;
    size_t i,j,insize;
    SV *subav, *subsubav; 
    pd = &(host_x[0]);
    av = (AV *)SvRV(x);
    insize = head->input_size;

    for(i=0;i<insize;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<1;j++){ // for the cols of that row

          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
    // now transfer to device
    gpu_memcpy_to_device(host_x, device_x, insize*sizeof(float));
    return 1;
}

int load_target(SV *y)
{
// outsize x 1 input array
    AV *av;
    float *pd; 
    size_t i,j,outsize;
    SV *subav, *subsubav; 
    pd = &(host_y[0]);
    av = (AV *)SvRV(y);
    outsize = tail->output_size;
    for(i=0;i<outsize;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<1;j++){ // for the cols of that row
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
    // now transfer to device
    gpu_memcpy_to_device(host_y, device_y, outsize*sizeof(float));
    return 1;
}  


void run_feed_forward() {
    node_t * current = head;
    
    float *activation;
    activation = device_x;
    run_gpu_transpose_2D_array(activation, device_x_transposed, head->input_size,1);
    while (current != NULL) {
        run_gpu_linear( activation, current->device_Weights, current->device_Bias, current->device_Output, current->input_size, current->output_size );
        gpu_sigmoid( current->device_Output, current->device_Activated_Output, current->output_size );
        activation = current->device_Activated_Output;
        current = current->next;
    }
}

void calculate_loss() {
   calculate_cost_and_derivative(tail->device_Activated_Output, device_y, device_Cost_Derivative, tail->output_size, 1);
}

void run_back_propagation() {
   node_t *current = tail;

   //float *delta = device_Cost_Derivative;
   gpu_memcpy_intra_device( device_Cost_Derivative, current->device_Delta, current->output_size*sizeof(float));

   gpu_add_same_size( current->device_Bias_Derivative, current->device_Delta, current->output_size );
   run_gpu_transpose_2D_array(current->prev->device_Activated_Output, current->prev->device_Activated_Output_Transposed, current->prev->output_size,1);
   run_gpu_weight_derivative( current->device_Delta, current->prev->device_Activated_Output_Transposed, current->device_Weights_Derivative, current->output_size,  1,current->prev->output_size ); 
   current = current->prev;
   while (current != NULL) {
// do the back prop
      run_gpu_sigmoid_prime(current->device_Activated_Output, current->device_Activated_Output_Derivative,current->output_size);
      float *sp = current->device_Activated_Output_Derivative;
      run_gpu_transpose_2D_array(current->next->device_Weights, current->next->device_Weights_Transposed, current->next->output_size, current->next->input_size);
      run_gpu_derivative(current->next->device_Weights_Transposed, current->next->device_Delta, sp, current->device_Delta, current->output_size, current->next->output_size,1); // wT x delta * sp (e.g. 30x10 x 10x1 => 30 x 1 * 30 x 1)
      float *activation;
      if (current->prev != NULL) {
         run_gpu_transpose_2D_array(current->prev->device_Activated_Output, current->prev->device_Activated_Output_Transposed, current->input_size,1);
         activation = current->prev->device_Activated_Output_Transposed;
      } else {
         activation = device_x_transposed;
      }
      run_gpu_weight_derivative( current->device_Delta, activation, current->device_Weights_Derivative, current->output_size, 1, current->input_size); // current->delta x next Activation Transposed and add the result to current weight derivative
      current = current->prev;
   }
}

void run_update_weights_and_biases(float modifier) {
   node_t * current = head;

   while (current != NULL) {
      run_gpu_update_weights(modifier, current);
      run_gpu_update_biases(modifier, current);
      current = current->next;
   }
}

int get_last_activated_output( SV *R ) {
  
    // Transfer results from host to perl
    AV *av, *av2;
    float *pd;
    size_t i,j,RH,RW, asz;
    
    RW = 1;
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
    return 0;
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

sub argmax {
   my $arr = shift; 
   my $max = $arr->[0][0];
   my $idx = [0,0];
   foreach my $i (0 .. $#{$arr}) {
      foreach my $j (0 .. $#{$arr->[0]}) {
         if ($arr->[$i][$j] > $max) {
            $max = $arr->[$i][$j];
            $idx = [$i, $j];
         }
      }
   }
   return $idx;
}   


sub create_network {
   my ($sizes, @options) = @_;
   my %params = @options;
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
            push @$iw, [random_normal( $sizes->[$i] ) ];
         }
         $params{weights}->[$i] = $iw;
      }
      return 0 unless add_node($sizes->[$i], $sizes->[$i + 1], $params{bias}->[$i], $params{weights}->[$i]);
   }
   # reserve RAM for initial input
   reset_derivatives();
   return 0 unless reserve_input_memory($sizes->[0], $sizes->[-1]);
   return 1;
}    

sub backprop {
   run_back_propagation();
}

sub feedforward {
   my $xy = shift;
   # convert input to C, put into already reserved memory
   return unless load_input($xy->[0]); 
   return unless load_target($xy->[1]); 
   # run the forward pass of the network
   run_feed_forward();
   #my $last_activated_output = [];
   #get_last_activated_output($last_activated_output) ;
   #say Dumper($last_activated_output);
}

sub update_weights {
  # run at end of mini batch
  # don't forget to zero the derivatives!
  my %params = @_;
  $params{batch_size} ||= 10;
  $params{learning_rate} ||= 3;
  run_update_weights_and_biases( $params{learning_rate} / $params{batch_size} );
  reset_derivatives();
}

sub update_mini_batch {
   my ($mb, $eta, $j, $ctr) = @_;
   foreach my $row (@$mb) {
      feedforward($row);
      calculate_loss();
      backprop();
   }   
   update_weights( batch_size => scalar(@$mb), learning_rate => $eta );
}

sub SGD {
   my $training_data = shift;
   my $epochs = shift;
   my $mini_batch_size = shift;
   my $eta = shift;
   my %params = @_;
   my $test_data = $params{test_data};
   my $n = scalar(@$training_data);
   my $n_test;
   if ($test_data) {
      $n_test = scalar(@$test_data);
   }
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
         if ($ctr % 50 == 0) { print "." };
         update_mini_batch($mb, $eta, $j, $ctr);
         $ctr++;
      }
      print "\n";
      if ($test_data) {
         say "Epoch $j : " . evaluate($test_data) . " / $n_test";
      } else {
         say "Epoch $j complete";
      }
      say "epoch time = " . tv_interval($start , [gettimeofday]);
   }
}

sub validation_feedforward {
   my $xy = shift;
   # convert input to C, put into already reserved memory
   return unless load_input($xy->[0]);
   return unless load_target($xy->[1]);
   # run the forward pass of the network
   run_feed_forward();
   my $last_activated_output = [];
   get_last_activated_output($last_activated_output) ;
   return $last_activated_output;
}

sub evaluate {
   my $data = shift;
   my $successes = 0;
   foreach my $d (@$data) {
      my $calc = validation_feedforward($d);

      my $expected = argmax($d->[1]);
      my $calculated = argmax($calc);
      if ($expected->[0] == $calculated->[0] and $expected->[1] == $calculated->[1]) {
         $successes++; 
      }
   }
   return $successes;
}

1;
