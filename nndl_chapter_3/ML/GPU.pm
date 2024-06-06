use Modern::Perl;
package ML::GPU;
use Math::Matrix;
use Math::Random;
use Data::Dumper;
use List::Util qw/shuffle/;
use Time::HiRes qw(gettimeofday tv_interval);
use Storable qw(dclone);
use File::Slurp;
use JSON;


require Exporter;
our @ISA = qw(Exporter);
our @EXPORT = qw(create_network print_network feedforward calculate_loss backprop update_weights SGD save_network load_network mnist_batch_guess mnist_image_guess);

my $code;

BEGIN {
        $code = <<'EOCODE';
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

// CUDA_CHECK translated from ROCM Common/example_utils.hpp
constexpr int error_exit_code = -1;

int debug = 0;
int loss_function = 1;

#define CUDA_CHECK(condition)                                                                \
    {                                                                                       \
        const cudaError_t error = condition;                                                 \
        if(error != cudaSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << cudaGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(error_exit_code);                                                     \
        }                                                                                   \
    }

typedef struct node {
    int input_size;
    int output_size;
    struct node * next;
    struct node * prev;
    float *host_Bias;
    float *device_Bias;
    float *host_Bias_Derivative;
    float *device_Bias_Derivative;
    float *host_Weights_Derivative;
    float *device_Weights_Derivative;
    SV *perl_Bias;
    float *host_Weights;
    float *device_Weights;
    SV *perl_Weights;
    float *host_Output;
    float *device_Output;
    float *host_Activated_Output;
    float *device_Activated_Output;
    float *host_Activated_Output_Derivative;
    float *device_Activated_Output_Derivative;
    float *host_Activated_Output_Transposed;
    float *device_Activated_Output_Transposed;
    float *host_Weights_Transposed;
    float *device_Weights_Transposed;
    float *host_Delta;
    float *device_Delta;
} node_t;

int   mini_batch_size;

#define BLOCK_SIZE 10

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
///////////////////////////////////
//
// gpu kernel section
//

__global__ void gpu_transpose_2D_array(float *in, float *transposed, size_t rows, int cols);

__global__ void gpu_transpose_2D_array(float *in, float *transposed, size_t rows, size_t cols)
{
	
   unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
   
   if (xIndex < cols && yIndex < rows)
   {
       unsigned int index_in  = xIndex + cols * yIndex;
       unsigned int index_out = yIndex + rows * xIndex;
       transposed[index_out] = in[index_in]; 
   }
}

__global__
void gpu_linear(float *a, float *b,  float *c, float *r, int m, int n, int k);

__global__ void gpu_linear(float *a, float *b,  float *c, float *r, int m, int n, int k)
{
// linear = weights x activation + bias, so r = a x b + c (using the short param names)
// weights = output x input (in this method weight indexing is reversed)
// activation = input x batch_size
// w x a = output x batch_size = m x n X n x k = m x k
// bias = output x 1, so bias needs to be broadcast
// result = 
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m)
    {
        for(size_t i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        r[row * k + col] = sum + c[row]; // bias is only 1 col wide
    }
}

__global__
void gpu_matmul(float *a, float *b,  float *r, int m, int n, int k);

__global__ void gpu_matmul(float *a, float *b,  float *r, int m, int n, int k)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m)
    {
        for(size_t i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        r[row * k + col] = sum ;
    }
}

__global__
void gpu_derivative(float *a, float *b,  float *c, float *r, int m, int n, int k);

__global__ void gpu_derivative(float *a, float *b,  float *c, float *r, int m, int n, int k)
{
// linear = weights x activation + bias, so r = a x b + c (using the short param names)
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m)
    {
        for(size_t i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        r[row * k + col] = sum * c[row * k + col];
    }
}

__global__ void gpu_update_weights(float modifier, float decay, float *a,float *b, size_t m, size_t n);

__global__ void gpu_update_weights(float modifier, float decay, float *a,float *b, size_t m, size_t n) {
 
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        a[row * n + col] = decay * a[row * n + col ] - modifier * b[row * n + col];
    }
}

__global__ void gpu_update_biases(float modifier, float *a,float *b, size_t m, size_t n);

__global__ void gpu_update_biases(float modifier, float *a,float *b, size_t m, size_t n) {
 
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < 1 && row < m)
    {
        for(size_t i = 0; i < n; i++)
        {
            sum += b[row * n + i];
        }
        a[row] -= modifier * sum;
    }
}

__global__
void gpu_weight_derivative(float *a, float *b, float *r, int m, int n, int k);

__global__ void gpu_weight_derivative(float *a, float *b, float *r, int m, int n, int k)
{
// weight_prime = delta x activation + current weight_prime, so r = a x b + r (using the short param names)
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m)
    {
        for(size_t i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        r[row * k + col] = sum + r[row * k + col];
    }
}

__global__ 
void gpu_matrix_sigmoid(float *a, float *b, int m, int n );

__global__ void gpu_matrix_sigmoid(float *a,float *b, int m, int n )
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        b[row * n + col] = 1 / ( 1 + exp( -1 * a[row * n + col] ) );
    }
}


__global__ void gpu_mse_cost(float *a, float *t, float *o, int m, int n );

__global__ void gpu_mse_cost(float *a, float *t, float *o, int m, int n ) {
   
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        o[row * n + col] = powf(a[row * n + col] - t[row * n + col],2)/2;
    }
}

__global__ void gpu_cle_cost(float *a, float *t, float *o, int m, int n );

__global__ void gpu_cle_cost(float *a, float *t, float *o, int m, int n ) {
   
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        float arg1 = a[row * n + col];
        if (arg1 > 0) {
           arg1 = log(a[row * n + col]);
        } else {
           arg1 = 0;
        }
        float arg2 = 1 - a[row * n + col];
        if (arg2 > 0) {
           arg2 = log(1 - a[row * n + col]);
        } else {
           arg2 = 0;
        }
        o[row * n + col] = -t[row * n + col]*arg1-(1-t[row * n + col])*arg2;
    }
}


__global__ void gpu_mse_cost_derivative(float *a, float *t, float *o, int m, int n );

__global__ void gpu_mse_cost_derivative(float *a, float *t, float *o, int m, int n ) {
   
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        o[row * n + col] = (a[row * n + col] - t[row * n + col]) * ( a[row * n + col] * ( 1 - a[row * n + col] ) );
    }
}

__global__ void gpu_cle_cost_derivative(float *a, float *t, float *o, int m, int n );

__global__ void gpu_cle_cost_derivative(float *a, float *t, float *o, int m, int n ) {
   
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        o[row * n + col] = (a[row * n + col] - t[row * n + col]);
    }
}

__global__ void gpu_sigmoid_prime(float *a, float *o, int m, int n );

__global__ void gpu_sigmoid_prime(float *a, float *o, int m, int n ) {
   
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        o[row * n + col] =  a[row * n + col] * ( 1 - a[row * n + col] ) ;
    }
}

__global__ void gpu_add_two_same_size(float *a,float *b, size_t m, size_t n);

__global__ void gpu_add_two_same_size(float *a,float *b, size_t m, size_t n) {
 
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        a[row * n + col] += b[row * n + col];
    }
}

// end of gpu kernel section
///////////////////////////////////

///////////////////////////////////
//
// gpu kernel wrapper functions
//

void print_2D_array(float *foo, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%+.5f\t", foo[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int run_gpu_linear( float *activation, float *device_Weights,  float *device_Bias, float *device_Output, int m, int n, int k) {
// m x n matmul n x k = m x k
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGridT(grid_cols, grid_rows);
    dim3 dimBlockT(BLOCK_SIZE, BLOCK_SIZE);
    gpu_linear<<<dimGridT, dimBlockT>>>(device_Weights, activation, device_Bias, device_Output,m, n, k);
    return 1;
}

int run_gpu_matmul( float *a, float *b,  float *r, int input_size, int middle_size, int output_size) {
    unsigned int grid_rows = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGridT(grid_cols, grid_rows);
    dim3 dimBlockT(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matmul<<<dimGridT, dimBlockT>>>(a, b, r, input_size,middle_size, output_size);
    return 1;
}

int run_gpu_derivative( float *a, float *b,  float *sp, float *r, int input_size, int middle_size, int output_size) {
    unsigned int grid_rows = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGridT(grid_cols, grid_rows);
    dim3 dimBlockT(BLOCK_SIZE, BLOCK_SIZE);
    gpu_derivative<<<dimGridT, dimBlockT>>>(a, b, sp, r,input_size, middle_size, output_size);
    return 1;
}

int run_gpu_weight_derivative( float *a, float *b, float *r, int input_size, int middle_size, int output_size) {
    unsigned int grid_rows = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGridT(grid_cols, grid_rows);
    dim3 dimBlockT(BLOCK_SIZE, BLOCK_SIZE);
    gpu_weight_derivative<<<dimGridT, dimBlockT>>>(a, b, r,input_size, middle_size, output_size);
    return 1;
}

int gpu_sigmoid( float *device_Output, float *device_Activated_Output, int output_size ) {
    unsigned int grid_rows = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (mini_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_matrix_sigmoid<<<dimGrid, dimBlock>>>(device_Output, device_Activated_Output, output_size, mini_batch_size);
    return 1;
}

int run_gpu_sigmoid_prime( float *device_Activated_Output, float *device_Activated_Output_Derivative, int output_size ) {
    unsigned int grid_rows = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (mini_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_sigmoid_prime<<<dimGrid, dimBlock>>>(device_Activated_Output, device_Activated_Output_Derivative, output_size, mini_batch_size);
    return 1;
}

int gpu_add_same_size( float *lhs, float *rhs, int arraysize ) {
    unsigned int grid_rows = (arraysize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_add_two_same_size<<<dimGrid, dimBlock>>>(lhs, rhs, arraysize, 1);
    return 1;
}
   
void run_gpu_transpose_2D_array( float *in, float *transpose, size_t rows, size_t cols) {
    unsigned int grid_rows = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_transpose_2D_array<<<dimGrid, dimBlock>>>(in, transpose, rows, cols);
}

void run_gpu_update_weights( float modifier, node_t *current, float decay) {
    unsigned int grid_rows = (current->output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (current->input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_update_weights<<<dimGrid, dimBlock>>>(modifier, decay, current->device_Weights, current->device_Weights_Derivative, current->output_size, current->input_size);
}
    
void run_gpu_update_biases( float modifier, node_t *current) {
    unsigned int grid_rows = (current->output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (mini_batch_size *  BLOCK_SIZE - 1) / BLOCK_SIZE; 
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_update_biases<<<dimGrid, dimBlock>>>(modifier, current->device_Bias, current->device_Bias_Derivative, current->output_size, mini_batch_size);
}

// end of gpu kernel wrapper section
///////////////////////////////////


node_t * head = NULL;
node_t * tail = NULL;

// these are where the initial input for each "feedforward" will be stored
float *host_x; 
float *host_x_transposed; 
float *device_x; 
float *device_x_transposed; 
SV    *perl_x; 
// these are where the target for each "feedforward" will be stored
float *host_y; 
float *device_y; 
float *host_y_transposed; 
float *device_y_transposed; 
SV    *perl_y; 

void reset_derivatives() {
    node_t * current = head;

    while (current != NULL) {
       for (int i = 0; i < current->output_size; ++i) {
           for (int j = 0; j < current->input_size; ++j) {
              current->host_Weights_Derivative[i *  current->input_size + j] = 0;
           }
           current->host_Bias_Derivative[i] = 0;
        }
        CUDA_CHECK(cudaMemcpy(current->device_Bias_Derivative, current->host_Bias_Derivative, current->output_size*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(current->device_Weights_Derivative, current->host_Weights_Derivative, current->input_size*current->output_size*sizeof(float), cudaMemcpyHostToDevice));
        current = current->next;
    }
}   

void print_list() {
    node_t * current = head;

    while (current != NULL) {
        printf("Input: %d, Output: %d\n", current->input_size, current->output_size);
        current = current->next;
    }
}


float calculate_cost(){
    // (output - target) * output_derivative : output_derivative can be derived from output, so do it all within the GPU
    unsigned int grid_rows = (tail->output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (mini_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    if (loss_function == 2) {
       gpu_cle_cost<<<dimGrid, dimBlock>>>( tail->device_Activated_Output, device_y, device_Cost, tail->output_size, mini_batch_size );
    } else {
       gpu_mse_cost<<<dimGrid, dimBlock>>>( tail->device_Activated_Output, device_y, device_Cost, tail->output_size, mini_batch_size );
    }


    if (debug == 1) {
       CUDA_CHECK(cudaMemcpy(tail->host_Activated_Output, tail->device_Activated_Output, mini_batch_size * tail->output_size * sizeof(float), cudaMemcpyDeviceToHost));
       std::cout << "final layer activation" << std::endl;
       print_2D_array(tail->host_Activated_Output, tail->output_size, mini_batch_size);
       std::cout << "targets" << std::endl;
       print_2D_array(host_y, tail->output_size, mini_batch_size);
   }
   CUDA_CHECK(cudaMemcpy(host_Cost, device_Cost, mini_batch_size * tail->output_size * sizeof(float), cudaMemcpyDeviceToHost));
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
         CUDA_CHECK(cudaMemcpy(current->host_Weights, current->device_Weights, current->input_size * current->output_size * sizeof(float), cudaMemcpyDeviceToHost));
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

int calculate_cost_derivative() {
    // (output - target) * output_derivative : output_derivative can be derived from output, so do it all within the GPU
    unsigned int grid_rows = (tail->output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (mini_batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    if (loss_function == 2) {
       gpu_cle_cost_derivative<<<dimGrid, dimBlock>>>( tail->device_Activated_Output, device_y, device_Cost_Derivative, tail->output_size, mini_batch_size );
    } else {
       gpu_mse_cost_derivative<<<dimGrid, dimBlock>>>( tail->device_Activated_Output, device_y, device_Cost_Derivative, tail->output_size, mini_batch_size );
    }


    if (debug == 1) {
       CUDA_CHECK(cudaMemcpy(tail->host_Activated_Output, tail->device_Activated_Output, mini_batch_size * tail->output_size * sizeof(float), cudaMemcpyDeviceToHost));
       std::cout << "final layer activation" << std::endl;
       print_2D_array(tail->host_Activated_Output, tail->output_size, mini_batch_size);
       CUDA_CHECK(cudaMemcpy(host_Cost_Derivative, device_Cost_Derivative, mini_batch_size * tail->output_size * sizeof(float), cudaMemcpyDeviceToHost));
       std::cout << "cost derivative calc" << std::endl;
       print_2D_array(host_Cost_Derivative, mini_batch_size, tail->output_size);
    }

    return 0;
}

void run_backpropagation() {
   node_t *current = tail;

   //float *delta = device_Cost_Derivative; 
   CUDA_CHECK(cudaMemcpy(current->device_Delta, device_Cost_Derivative, mini_batch_size * current->output_size*sizeof(float), cudaMemcpyDeviceToDevice));
   CUDA_CHECK(cudaMemcpy(current->device_Bias_Derivative, device_Cost_Derivative, mini_batch_size * current->output_size*sizeof(float), cudaMemcpyDeviceToDevice));
   //gpu_add_same_size( current->device_Bias_Derivative, current->device_Delta, mini_batch_size, current->output_size );
   if (debug == 1) {
      CUDA_CHECK(cudaMemcpy(current->prev->host_Activated_Output, current->prev->device_Activated_Output, tail->input_size*mini_batch_size*sizeof(float), cudaMemcpyDeviceToHost));
      std::cout << "Initial Backpass Input Activations " << std::endl;
      print_2D_array(current->prev->host_Activated_Output,  tail->input_size, mini_batch_size); 
   }
   run_gpu_transpose_2D_array(current->prev->device_Activated_Output, current->prev->device_Activated_Output_Transposed,  current->prev->output_size, mini_batch_size);
   if (debug == 1) {
      CUDA_CHECK(cudaMemcpy(current->prev->host_Activated_Output_Transposed, current->prev->device_Activated_Output_Transposed, tail->input_size*mini_batch_size*sizeof(float), cudaMemcpyDeviceToHost));
      std::cout << "Initial Backpass Input Activations Transposed" << std::endl;
      print_2D_array(current->prev->host_Activated_Output_Transposed, mini_batch_size, tail->input_size ); // it's transposed so now outsize x rows, rather than rows x outsize
      printf("host_Delta %p\n", current->host_Delta);
      printf("device_Delta %p\n", current->device_Delta);
      printf("outsize %d\n", tail->output_size);
      CUDA_CHECK(cudaMemcpy(current->host_Delta, current->device_Delta, tail->output_size*mini_batch_size*sizeof(float), cudaMemcpyDeviceToHost));
      std::cout << "Initial Backpass Input Delta" << std::endl;
      print_2D_array(current->host_Delta, tail->output_size, mini_batch_size);
   }
   run_gpu_weight_derivative( current->device_Delta , current->prev->device_Activated_Output_Transposed,  current->device_Weights_Derivative, current->output_size, mini_batch_size, current->input_size );
   if (debug == 1) {
      CUDA_CHECK(cudaMemcpy(current->host_Weights_Derivative, current->device_Weights_Derivative, current->input_size*current->output_size*sizeof(float), cudaMemcpyDeviceToHost));
      std::cout << "Last Layer Backpass Weights Derivative" << std::endl;
      print_2D_array(current->host_Weights_Derivative, current->output_size, current->input_size);
   }

   current = current->prev;
   while (current != NULL) {
// do the back prop
      run_gpu_sigmoid_prime(current->device_Activated_Output, current->device_Activated_Output_Derivative,current->output_size);
      if (debug == 1) {
         printf("host_Activated_Output_Derivative %p\n", current->host_Activated_Output_Derivative);
         printf("device_Activated_Output_Derivative %p\n", current->device_Activated_Output_Derivative);
         printf("outsize %d\n", current->output_size);
         CUDA_CHECK(cudaMemcpy(current->host_Activated_Output_Derivative, current->device_Activated_Output_Derivative, mini_batch_size*current->output_size*sizeof(float), cudaMemcpyDeviceToHost));
         std::cout << "Activated Output Derivative" << std::endl;
         print_2D_array(current->host_Activated_Output_Derivative, current->output_size, mini_batch_size);
      }
      float *sp = current->device_Activated_Output_Derivative;
      run_gpu_transpose_2D_array(current->next->device_Weights, current->next->device_Weights_Transposed, current->next->output_size, current->next->input_size);
      if (debug == 1) {
         CUDA_CHECK(cudaMemcpy(current->next->host_Weights_Transposed, current->next->device_Weights_Transposed, current->next->input_size*current->next->output_size*sizeof(float), cudaMemcpyDeviceToHost));
         std::cout << "Weights (next) transposed" << std::endl;
         print_2D_array(current->next->host_Weights_Transposed, current->next->input_size, current->next->output_size);
      }
      run_gpu_derivative(current->next->device_Weights_Transposed, current->next->device_Delta, sp, current->device_Delta, current->output_size, current->next->output_size,mini_batch_size); // wT x delta * sp (e.g. 30x10 x 10x1 => 30 x 1 * 30 x 1)
      if (debug == 1) {
         CUDA_CHECK(cudaMemcpy(current->host_Delta, current->device_Delta, mini_batch_size*current->output_size*sizeof(float), cudaMemcpyDeviceToHost));
         std::cout << "New delta" << std::endl;
         print_2D_array(current->host_Delta, current->output_size, mini_batch_size);
      }
      float *activation;
      if (current->prev != NULL) {
         run_gpu_transpose_2D_array(current->prev->device_Activated_Output, current->prev->device_Activated_Output_Transposed, current->prev->output_size,mini_batch_size);
         activation = current->prev->device_Activated_Output_Transposed;
      } else {
         activation = device_x_transposed;
      }   
      CUDA_CHECK(cudaMemcpy(current->device_Bias_Derivative, current->device_Delta, mini_batch_size * current->output_size*sizeof(float), cudaMemcpyDeviceToDevice));
      run_gpu_matmul(current->device_Delta, activation, current->device_Weights_Derivative, current->output_size, mini_batch_size, current->input_size);
      if (debug == 1) {
         CUDA_CHECK(cudaMemcpy(current->host_Weights_Derivative, current->device_Weights_Derivative, current->output_size*current->input_size*sizeof(float), cudaMemcpyDeviceToHost));
         std::cout << "New Weights Derivative" << std::endl;
         print_2D_array(current->host_Weights_Derivative, current->output_size, current->input_size);
      }
      current = current->prev;
   }
}
   
void run_feed_forward() {
    node_t * current = head;

    float *activation;
    activation = device_x;
    while (current != NULL) {
        run_gpu_linear( activation, current->device_Weights, current->device_Bias, current->device_Output, current->output_size, current->input_size, mini_batch_size );
        if (debug == 1) {
           if (current == head) {
              CUDA_CHECK(cudaMemcpy(host_x, device_x, head->input_size*mini_batch_size*sizeof(float), cudaMemcpyDeviceToHost));
              std::cout << "Initial Input Activations" << std::endl;
              print_2D_array(host_x, head->input_size, mini_batch_size);
           } else {
              CUDA_CHECK(cudaMemcpy(current->prev->host_Activated_Output, current->prev->device_Activated_Output, current->input_size*mini_batch_size*sizeof(float), cudaMemcpyDeviceToHost));
              std::cout << "Input Activations" << std::endl;
              print_2D_array(current->prev->host_Activated_Output, current->input_size, mini_batch_size);
           }
           CUDA_CHECK(cudaMemcpy(tail->host_Bias, tail->device_Bias, tail->output_size*sizeof(float), cudaMemcpyDeviceToHost));
           std::cout << "Bias" << std::endl;
           print_2D_array(current->host_Bias, current->output_size, 1);
           CUDA_CHECK(cudaMemcpy(tail->host_Weights, tail->device_Weights, tail->input_size*tail->output_size*sizeof(float), cudaMemcpyDeviceToHost));
           std::cout << "Weights" << std::endl;
           print_2D_array(current->host_Weights, current->output_size, current->input_size);
           CUDA_CHECK(cudaMemcpy(current->host_Output, current->device_Output, mini_batch_size * current->output_size * sizeof(float), cudaMemcpyDeviceToHost));
           std::cout << "Output before activation" << std::endl;
           print_2D_array(current->host_Output, current->output_size, mini_batch_size);
        }
        gpu_sigmoid( current->device_Output, current->device_Activated_Output, current->output_size );
        if (debug == 1) {
           CUDA_CHECK(cudaMemcpy(current->host_Activated_Output, current->device_Activated_Output, mini_batch_size * current->output_size * sizeof(float), cudaMemcpyDeviceToHost));
           std::cout << "Output after activation" << std::endl;
           print_2D_array(current->host_Activated_Output,  current->output_size, mini_batch_size );
        }
        activation = current->device_Activated_Output;
        current = current->next;
    }
}

int load_input(SV *x, int elements) 
{
// "batch rows x insize" input array
    AV *av;
    float *pd;
    size_t i,j,insize;
    SV *subav, *subsubav; 
    mini_batch_size = elements;
    if (debug == 1) {
       size_t AH, AW, *AWs = NULL;
       array_numelts_2D(x, &AH, &AWs);
       AW = AWs[0];
       std::cout << "load_input: AH (rows) = " << AH << " AW (cols) = " << AW << std::endl;
    }
// the data coming from Perl is batch size x input size, but it needs to be input size x batch size (i.e. transposed) for "linear" to work.
// linear is w x a + b, where w is output x input, and b will be 1 x output size, which needs be broadcast against all rows in the batch.
// the result of w x a should be output x batch size, and b will be added 1 column at a time.
// In the demo: w is 30 x 784, and a is 784 x 10 (i.e. transposed from the Perl input of 10 x 784), so the result is a 30 x 10 matrix.
// b is 1 x 30 (i.e. 1 row of 30 biases) and this needs to be added added 1 column at a time (since there are 10 columns, 1 per entry of the mini batch)

    pd = &(host_x_transposed[0]);
if (debug == 1) {
   printf("pd = %p\n", pd);
}
    av = (AV *)SvRV(x);
    insize = head->input_size;
    for(i=0;i<elements;i++){ // for each row
       subav = *av_fetch(av, i, FALSE);
       for(j=0;j<insize;j++){ // for the cols of that row
if (debug == 1) {
   std::cout << "i,j = " << i << "," << j << std::endl;
}
          subsubav = *av_fetch((AV *)SvRV(subav), j, FALSE);
          *pd = SvNV(subsubav);
          pd++;
       }
    }
    if (debug == 1) {
       std::cout << "load_input perl -> c complete" << std::endl;
    }
    // now transfer to device
    CUDA_CHECK(cudaMemcpy(device_x_transposed, host_x_transposed, mini_batch_size*insize*sizeof(float), cudaMemcpyHostToDevice));
    run_gpu_transpose_2D_array(device_x_transposed, device_x, mini_batch_size, insize);
    if (debug == 1) {
       CUDA_CHECK(cudaMemcpy(host_x, device_x, insize*mini_batch_size*sizeof(float), cudaMemcpyDeviceToHost));
       print_2D_array(host_x, insize, mini_batch_size);
    }

    return 1;
}

int load_target(SV *y)
{
// "batch rows x outsize" output array
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
    CUDA_CHECK(cudaMemcpy(device_y_transposed, host_y_transposed, mini_batch_size*outsize*sizeof(float), cudaMemcpyHostToDevice));
    run_gpu_transpose_2D_array(device_y_transposed, device_y, mini_batch_size, outsize);
    if (debug == 1) {
       CUDA_CHECK(cudaMemcpy(host_y, device_y, outsize*mini_batch_size*sizeof(float), cudaMemcpyDeviceToHost));
       print_2D_array(host_y, outsize, mini_batch_size);
    }
    return 1;
}  

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
    if (debug == 1) {
       std::cout << "outsize = " << outsize << std::endl;
       printf("host_Bias = %p\n", new_node->host_Bias);
    }
    CUDA_CHECK(cudaMallocHost((void **) &new_node->host_Bias, sizeof(float)*1*outsize)); // bias is 1 x outsize
    CUDA_CHECK(cudaMalloc((void **) &(new_node->device_Bias), sizeof(float)*1*outsize)); // bias is 1 x outsize
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
    CUDA_CHECK(cudaMemcpy(new_node->device_Bias, new_node->host_Bias, 1*outsize*sizeof(float), cudaMemcpyHostToDevice));
// convert Perl weight array to C array of floats and push it onto the GPU
    CUDA_CHECK(cudaMallocHost((void **) &new_node->host_Weights, sizeof(float)*insize*outsize)); // weights are insize x outsize
    CUDA_CHECK(cudaMalloc((void **) &new_node->device_Weights, sizeof(float)*insize*outsize)); // weights are insize x outsize
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
    CUDA_CHECK(cudaMemcpy(new_node->device_Weights, new_node->host_Weights, insize*outsize*sizeof(float), cudaMemcpyHostToDevice));
// reserve memory for output and activated output (both 1 x outsize)
    CUDA_CHECK(cudaMallocHost((void **) &new_node->host_Output, sizeof(float)*outsize*batch_size)); 
    CUDA_CHECK(cudaMalloc((void **) &new_node->device_Output, sizeof(float)*outsize*batch_size)); 
    CUDA_CHECK(cudaMallocHost((void **) &new_node->host_Activated_Output, sizeof(float)*outsize*batch_size)); 
    CUDA_CHECK(cudaMalloc((void **) &new_node->device_Activated_Output, sizeof(float)*outsize*batch_size)); 
    CUDA_CHECK(cudaMallocHost((void **) &new_node->host_Activated_Output_Derivative, sizeof(float)*outsize*batch_size)); 
    CUDA_CHECK(cudaMalloc((void **) &new_node->device_Activated_Output_Derivative, sizeof(float)*outsize*batch_size)); 
    CUDA_CHECK(cudaMallocHost((void **) &new_node->host_Activated_Output_Transposed, sizeof(float)*outsize*batch_size)); 
    CUDA_CHECK(cudaMalloc((void **) &new_node->device_Activated_Output_Transposed, sizeof(float)*outsize*batch_size)); 
// reserve memory for deriviatives, bias = 1 * outsize, weight = insize * outsize 
    CUDA_CHECK(cudaMallocHost((void **) &new_node->host_Weights_Derivative, sizeof(float)*insize*outsize)); // weights are insize x outsize
    CUDA_CHECK(cudaMalloc((void **) &new_node->device_Weights_Derivative, sizeof(float)*insize*outsize)); // weights are insize x outsize
    CUDA_CHECK(cudaMallocHost((void **) &new_node->host_Bias_Derivative, sizeof(float)*batch_size*outsize)); // bias is 1 x outsize
    CUDA_CHECK(cudaMalloc((void **) &(new_node->device_Bias_Derivative), sizeof(float)*batch_size*outsize)); // bias is 1 x outsize
// reserve memory for transposed Weights
    CUDA_CHECK(cudaMallocHost((void **) &new_node->host_Weights_Transposed, sizeof(float)*insize*outsize)); // weights are insize x outsize
    CUDA_CHECK(cudaMalloc((void **) &new_node->device_Weights_Transposed, sizeof(float)*insize*outsize)); // weights are insize x outsize
// reserve memory for temporary derivative calculation
    CUDA_CHECK(cudaMallocHost((void **) &new_node->host_Delta, sizeof(float)*outsize*batch_size)); // delta is  batch x outsize
    CUDA_CHECK(cudaMalloc((void **) &new_node->device_Delta, sizeof(float)*outsize*batch_size)); 

    new_node->next = NULL; 
    new_node->prev = NULL; 
    return new_node; 
} 

int add_node(int insize, int outsize, SV *biases, SV *weights, int batch_size)
{
    node_t* new_node = create_node(insize, outsize, biases, weights, batch_size); 
    if (new_node == NULL) {
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

int reserve_input_memory(int insize, int outsize, int batch_size)
{
// memory for input array
    CUDA_CHECK(cudaMallocHost((void **) &host_x, sizeof(float)*insize * batch_size)); // initial input is 1 x insize
    CUDA_CHECK(cudaMallocHost((void **) &host_x_transposed, sizeof(float)*insize * batch_size)); // initial input is 1 x insize
    CUDA_CHECK(cudaMalloc((void **) &device_x, sizeof(float)*insize * batch_size));   // initial_input is 1 x insize
    CUDA_CHECK(cudaMalloc((void **) &device_x_transposed, sizeof(float)*insize * batch_size));   // initial_input is 1 x insize
// memory for target
    CUDA_CHECK(cudaMallocHost((void **) &host_y, sizeof(float)*outsize*batch_size)); // target is 1 x outsize
    CUDA_CHECK(cudaMalloc((void **) &device_y, sizeof(float)*outsize*batch_size));  // target is 1 x outsize
    CUDA_CHECK(cudaMallocHost((void **) &host_y_transposed, sizeof(float)*outsize*batch_size)); // target is 1 x outsize
    CUDA_CHECK(cudaMalloc((void **) &device_y_transposed, sizeof(float)*outsize*batch_size));  // target is 1 x outsize
// memory for cost derivative, 1 x outsize
    CUDA_CHECK(cudaMallocHost((void **) &host_Cost_Derivative, sizeof(float)*outsize*batch_size));
    CUDA_CHECK(cudaMalloc((void **) &device_Cost_Derivative, sizeof(float)*outsize*batch_size));
// memory for cost , 1 x outsize
    CUDA_CHECK(cudaMallocHost((void **) &host_Cost, sizeof(float)*outsize*batch_size));
    CUDA_CHECK(cudaMalloc((void **) &device_Cost, sizeof(float)*outsize*batch_size));
// set network batch size
    mini_batch_size = batch_size;
    return 1;
}


int get_last_activated_output( SV *R ) {

    // Transfer results from host to perl
    AV *av, *av2;
    float *pd;
    size_t i,j,RH,RW, asz;

    RW = mini_batch_size;
    RH = tail->output_size;

// copy device data back to the host before loading the Perl values

    CUDA_CHECK(cudaMemcpy(tail->host_Activated_Output, tail->device_Activated_Output, RW*RH*sizeof(float), cudaMemcpyDeviceToHost));

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
       CUDA_CHECK(cudaMemcpy(tail->prev->host_Activated_Output, tail->prev->device_Activated_Output, tail->input_size*mini_batch_size*sizeof(float), cudaMemcpyDeviceToHost));
       std::cout << "Input Activations" << std::endl;
       print_2D_array(tail->host_Activated_Output, mini_batch_size, tail->output_size);
       CUDA_CHECK(cudaMemcpy(tail->host_Bias, tail->device_Bias, tail->output_size*sizeof(float), cudaMemcpyDeviceToHost));
       std::cout << "Bias" << std::endl;
       print_2D_array(tail->host_Bias, 1, tail->output_size);
       CUDA_CHECK(cudaMemcpy(tail->host_Weights, tail->device_Weights, tail->input_size*tail->output_size*sizeof(float), cudaMemcpyDeviceToHost));
       std::cout << "Weights" << std::endl;
       print_2D_array(tail->host_Weights, tail->output_size, tail->input_size);
       CUDA_CHECK(cudaMemcpy(tail->host_Output, tail->device_Output, RW*RH*sizeof(float), cudaMemcpyDeviceToHost));
       std::cout << "Output before activation" << std::endl;
       print_2D_array(tail->host_Output, RH, RW);
       std::cout << "Activtated Output" << std::endl;
       print_2D_array(tail->host_Activated_Output, RH, RW);
    }
    return 0;
}

void run_update_weights_and_biases(float modifier, float decay) {
   node_t * current = head;

   while (current != NULL) {
      if (debug == 1) {
         CUDA_CHECK(cudaMemcpy(current->host_Weights, current->device_Weights, current->input_size*current->output_size*sizeof(float), cudaMemcpyDeviceToHost));
         std::cout << "Weights before update" << std::endl;
         print_2D_array(current->host_Weights, current->output_size, current->input_size);
         CUDA_CHECK(cudaMemcpy(current->host_Weights_Derivative, current->device_Weights_Derivative, current->input_size*current->output_size*sizeof(float), cudaMemcpyDeviceToHost));
         std::cout << "Weights_Derivative" << std::endl;
         print_2D_array(current->host_Weights_Derivative, current->output_size, current->input_size);
      }
      run_gpu_update_weights(modifier, current, decay); 
      if (debug == 1) {
         CUDA_CHECK(cudaMemcpy(current->host_Weights, current->device_Weights, current->input_size*current->output_size*sizeof(float), cudaMemcpyDeviceToHost));
         std::cout << "Updated Weights" << std::endl;
         print_2D_array(current->host_Weights, current->output_size, current->input_size);
         CUDA_CHECK(cudaMemcpy(current->host_Bias, current->device_Bias, current->output_size*sizeof(float), cudaMemcpyDeviceToHost));
         std::cout << "Biases before update" << std::endl;
         print_2D_array(current->host_Bias, current->output_size, 1);
         CUDA_CHECK(cudaMemcpy(current->host_Bias_Derivative, current->device_Bias_Derivative, mini_batch_size * current->output_size*sizeof(float), cudaMemcpyDeviceToHost));
         std::cout << "Bias Derivative" << std::endl;
         print_2D_array(current->host_Bias_Derivative, current->output_size, mini_batch_size);
      }
      run_gpu_update_biases(modifier, current);
      if (debug == 1) {
         CUDA_CHECK(cudaMemcpy(current->host_Bias, current->device_Bias, current->output_size*sizeof(float), cudaMemcpyDeviceToHost));
         std::cout << "Updated Biases" << std::endl;
         print_2D_array(current->host_Bias, current->output_size, 1);
      }
      current = current->next;
   }
   dirty_dirty_weights = 1;
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
   CUDA_CHECK(cudaMemcpy(current->host_Weights, current->device_Weights, current->output_size*current->input_size*sizeof(float), cudaMemcpyDeviceToHost));
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
   CUDA_CHECK(cudaMemcpy(current->host_Bias, current->device_Bias, current->output_size*sizeof(float), cudaMemcpyDeviceToHost));
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

use Inline CUDA => Config =>
        host_code_language => 'cpp',
        BUILD_NOISY => 1,
        clean_after_build => 0,
        warnings => 10,
;

use Inline CUDA => $code;

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
      #say "input size = " . $sizes->[$i];
      #say "output size = " . $sizes->[$i + 1];
      if (!defined($params{ bias }->[$i])) {
         # this will be an "output size" x 1 array
         $params{bias}->[$i] = Math::Matrix->new([random_normal($sizes->[$i + 1])])->transpose()->as_array(); 
      }
      if (!defined($params{ weights }->[$i])) {
         # this will be an "output size" x "input size" array
         my $iw = [];
         foreach my $s (1 .. $sizes->[$i + 1]) {
            push @$iw, [ map { $_ / $scale_factor } random_normal( $sizes->[$i] ) ];
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
   return unless load_input(\@x, $elements);  # could be 1 or up to "batch size" elements
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
   return unless load_input(\@x, $elements);  # could be 1 or "batch size" elements
   return unless load_target(\@y); 
   # run the forward pass of the network
   run_feed_forward();
   my $last_activated_output = [];
   get_last_activated_output($last_activated_output) ;
   return $last_activated_output;
}

sub calculate_loss {
   calculate_cost_derivative() and die;
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
foreach my $key (keys %$data) {
   say $key;
}
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
   my @batch;
   $batch[0]->[0] = $data; 
   $batch[0]->[1] = [(0) x 10]; # validation expects to see a target array, but it isn't needed, so just make it zeros.
   my $calc = validation_feedforward(\@batch);
   return argmax($calc);
}

1;
