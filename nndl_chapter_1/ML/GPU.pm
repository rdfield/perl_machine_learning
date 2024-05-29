use Modern::Perl;
package ML::GPU;
use Math::Matrix;
use Math::Random;
use Data::Dumper;
use List::Util qw/shuffle/;
use Time::HiRes qw(gettimeofday tv_interval);


require Exporter;
our @ISA = qw(Exporter);
our @EXPORT = qw(create_network print_network feedforward calculate_loss backprop show_final_weights update_weights SGD);

my $code;

BEGIN {
        $code = <<'EOCODE';
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

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

#define BLOCK_SIZE 16

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
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m)
    {
        for(size_t i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        r[row * k + col] = sum + c[row * k + col];
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

__global__ void gpu_update_weights(float modifier, float *a,float *b, size_t m, size_t n);

__global__ void gpu_update_weights(float modifier, float *a,float *b, size_t m, size_t n) {
 
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        a[row * n + col] -= modifier * b[row * n + col];
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

__global__ void gpu_sigmoid_cost_derivative(float *a, float *t, float *o, int m, int n );

__global__ void gpu_sigmoid_cost_derivative(float *a, float *t, float *o, int m, int n ) {
   
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < m)
    {
        o[row * n + col] = (a[row * n + col] - t[row * n + col]) * ( a[row * n + col] * ( 1 - a[row * n + col] ) );
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
printf("%p\n", (void *) &foo);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%+.5f\t", foo[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int run_gpu_linear( float *activation, float *device_Weights,  float *device_Bias, float *device_Output, int input_size, int output_size) {
    unsigned int grid_rows = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGridT(grid_cols, grid_rows);
    dim3 dimBlockT(BLOCK_SIZE, BLOCK_SIZE);
    gpu_linear<<<dimGridT, dimBlockT>>>(device_Weights, activation, device_Bias, device_Output,output_size, input_size, 1);
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
    unsigned int grid_cols = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_matrix_sigmoid<<<dimGrid, dimBlock>>>(device_Output, device_Activated_Output, output_size, 1);
    return 1;
}

int run_gpu_sigmoid_prime( float *device_Activated_Output, float *device_Activated_Output_Derivative, int output_size ) {
    unsigned int grid_rows = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_sigmoid_prime<<<dimGrid, dimBlock>>>(device_Activated_Output, device_Activated_Output_Derivative, output_size, 1);
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

void run_gpu_update_weights( float modifier, node_t *current) {
    unsigned int grid_rows = (current->output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (current->input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_update_weights<<<dimGrid, dimBlock>>>(modifier, current->device_Weights, current->device_Weights_Derivative, current->output_size, current->input_size);
}
    
void run_gpu_update_biases( float modifier, node_t *current) {
    unsigned int grid_rows = (current->output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_update_weights<<<dimGrid, dimBlock>>>(modifier, current->device_Bias, current->device_Bias_Derivative, current->output_size, 1);
}

// end of gpu kernel wrapper section
///////////////////////////////////


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

void reset_derivatives() {
    cudaError_t cudret;
    node_t * current = head;

    while (current != NULL) {
       for (int i = 0; i < current->output_size; ++i) {
           for (int j = 0; j < current->input_size; ++j) {
              current->host_Weights_Derivative[i *  current->input_size + j] = 0;
           }
           current->host_Bias_Derivative[i] = 0;
        }
        if( (cudret=cudaMemcpy(current->device_Bias_Derivative, current->host_Bias_Derivative, current->output_size*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ){
            fprintf(stderr, "reset_derivative() : error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for transfering Bias derivative input from host to device: %s\n", cudaGetErrorString(cudret));
            return;
        }
        if( (cudret=cudaMemcpy(current->device_Weights_Derivative, current->host_Weights_Derivative, current->input_size*current->output_size*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ){
            fprintf(stderr, "reset_derivative() : error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for transfering Weights derivative input from host to device: %s\n", cudaGetErrorString(cudret));
            return;
        }
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

int calculate_cost_and_derivative(SV *R) {
    // (output - target) * output_derivative : output_derivative can be derived from output, so do it all within the GPU
    unsigned int grid_rows = (tail->output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_sigmoid_cost_derivative<<<dimGrid, dimBlock>>>( tail->device_Activated_Output, device_y, device_Cost_Derivative, tail->output_size, 1 );

    return 0;
}

void run_backpropagation() {
   node_t *current = tail;
   cudaError_t cudret;

   //float *delta = device_Cost_Derivative; 
   if( (cudret=cudaMemcpy(current->device_Delta, device_Cost_Derivative, current->output_size*sizeof(float), cudaMemcpyDeviceToDevice)) != cudaSuccess ){
       fprintf(stderr, "run_backpropagation() : error, call to cudaMemcpy(cudaMemcpyDeviceToDevice) has failed for transfering cost derivative to delta: %s\n", cudaGetErrorString(cudret));
   } 
   gpu_add_same_size( current->device_Bias_Derivative, current->device_Delta, current->output_size );
   run_gpu_transpose_2D_array(current->prev->device_Activated_Output, current->prev->device_Activated_Output_Transposed, current->prev->output_size,1);
   run_gpu_weight_derivative( current->device_Delta, current->prev->device_Activated_Output_Transposed, current->device_Weights_Derivative, current->output_size,  1,current->prev->output_size );
   if( (cudret=cudaDeviceSynchronize()) != cudaSuccess ){
      fprintf(stderr, "run_backpropagation() : error, call to cudaDeviceSynchronize() has failed: %s\n", cudaGetErrorString(cudret));
   }
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
   
void run_feed_forward() {
    cudaError_t cudret;
    node_t * current = head;


    float *activation;
    activation = device_x;
    run_gpu_transpose_2D_array(activation, device_x_transposed, head->input_size,1);
    while (current != NULL) {
        run_gpu_linear( activation, current->device_Weights, current->device_Bias, current->device_Output, current->input_size, current->output_size );

        gpu_sigmoid( current->device_Output, current->device_Activated_Output, current->output_size );
        if( (cudret=cudaDeviceSynchronize()) != cudaSuccess ){
            fprintf(stderr, "run_feed_forward() : error, call to cudaDeviceSynchronize() has failed: %s\n", cudaGetErrorString(cudret));
        }
        activation = current->device_Activated_Output;

        current = current->next;
    }
}

int load_input(SV *x) 
{
// insize x 1 input array
    AV *av;
    float *pd;
    size_t i,j,insize;
    SV *subav, *subsubav; 
    cudaError_t cudret;
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
    if( (cudret=cudaMemcpy(device_x, host_x, 1*insize*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ){
       fprintf(stderr, "load_input() : error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for transfering initial input from host to device: %s\n", cudaGetErrorString(cudret));
       return 0;
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
    cudaError_t cudret;
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
    if( (cudret=cudaMemcpy(device_y, host_y, 1*outsize*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ){
       fprintf(stderr, "load_target() : error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for transfering initial target from host to device: %s\n", cudaGetErrorString(cudret));
       return 0;
    }
    return 1;
}  

node* create_node(int insize, int outsize, SV *biases, SV *weights) 
{ 
    cudaError_t cudret;
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
    if( (cudret=cudaMallocHost((void **) &new_node->host_Bias, sizeof(float)*1*outsize)) != cudaSuccess ){ // bias is 1 x outsize
        fprintf(stderr, "create_node() : error, call to cudaMallocHost() has failed for host_Bias for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*1*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMalloc((void **) &(new_node->device_Bias), sizeof(float)*1*outsize)) != cudaSuccess ){ // bias is 1 x outsize
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for device_Bias for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*1*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }
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
    if( (cudret=cudaMemcpy(new_node->device_Bias, new_node->host_Bias, 1*outsize*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ){
       fprintf(stderr, "create_node() : error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for transfering bias from host to device: %s\n", cudaGetErrorString(cudret));
       return NULL;
    }
// convert Perl weight array to C array of floats and push it onto the GPU
    if( (cudret=cudaMallocHost((void **) &new_node->host_Weights, sizeof(float)*insize*outsize)) != cudaSuccess ){ // weights are insize x outsize
        fprintf(stderr, "create_node() : error, call to cudaMallocHost() has failed for host_Weights for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*insize*outsize, outsize, insize, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMalloc((void **) &new_node->device_Weights, sizeof(float)*insize*outsize)) != cudaSuccess ){ // weights are insize x outsize
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for device_Weights for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*insize*outsize, outsize, insize, cudaGetErrorString(cudret));
        return NULL;
    }
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
    if( (cudret=cudaMemcpy(new_node->device_Weights, new_node->host_Weights, insize*outsize*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess ){
       fprintf(stderr, "create_node() : error, call to cudaMemcpy(cudaMemcpyHostToDevice) has failed for transfering weights from host to device: %s\n", cudaGetErrorString(cudret));
       return NULL;
    }
// reserve memory for output and activated output (both 1 x outsize)
    if( (cudret=cudaMallocHost((void **) &new_node->host_Output, sizeof(float)*outsize)) != cudaSuccess ){ 
        fprintf(stderr, "create_node() : error, call to cudaMallocHost() has failed for host_Output for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMalloc((void **) &new_node->device_Output, sizeof(float)*outsize)) != cudaSuccess ){ 
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for device_Output for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMallocHost((void **) &new_node->host_Activated_Output, sizeof(float)*outsize)) != cudaSuccess ){ 
        fprintf(stderr, "create_node() : error, call to cudaMallocHost() has failed for host_Activated_Output for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMalloc((void **) &new_node->device_Activated_Output, sizeof(float)*outsize)) != cudaSuccess ){ 
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for device_Activated_Output for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMallocHost((void **) &new_node->host_Activated_Output_Derivative, sizeof(float)*outsize)) != cudaSuccess ){ 
        fprintf(stderr, "create_node() : error, call to cudaMallocHost() has failed for host_Activated_Output_Derivative for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMalloc((void **) &new_node->device_Activated_Output_Derivative, sizeof(float)*outsize)) != cudaSuccess ){ 
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for device_Activated_Output_Derivative for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMalloc((void **) &new_node->host_Activated_Output_Transposed, sizeof(float)*outsize)) != cudaSuccess ){ 
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for host_Activated_Output_Transposed for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, 1, outsize, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMalloc((void **) &new_node->device_Activated_Output_Transposed, sizeof(float)*outsize)) != cudaSuccess ){ 
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for device_Activated_Output_Transposed for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, 1, outsize, cudaGetErrorString(cudret));
        return NULL;
    }
// reserve memory for deriviatives, bias = 1 * outsize, weight = insize * outsize 
    if( (cudret=cudaMallocHost((void **) &new_node->host_Weights_Derivative, sizeof(float)*insize*outsize)) != cudaSuccess ){ // weights are insize x outsize
        fprintf(stderr, "create_node() : error, call to cudaMallocHost() has failed for host_Weights for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*insize*outsize, outsize, insize, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMalloc((void **) &new_node->device_Weights_Derivative, sizeof(float)*insize*outsize)) != cudaSuccess ){ // weights are insize x outsize
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for device_Weights for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*insize*outsize, outsize, insize, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMallocHost((void **) &new_node->host_Bias_Derivative, sizeof(float)*1*outsize)) != cudaSuccess ){ // bias is 1 x outsize
        fprintf(stderr, "create_node() : error, call to cudaMallocHost() has failed for host_Bias for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*1*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMalloc((void **) &(new_node->device_Bias_Derivative), sizeof(float)*1*outsize)) != cudaSuccess ){ // bias is 1 x outsize
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for device_Bias for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*1*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }
// reserve memory for transposed Weights
    if( (cudret=cudaMallocHost((void **) &new_node->host_Weights_Transposed, sizeof(float)*insize*outsize)) != cudaSuccess ){ // weights are insize x outsize
        fprintf(stderr, "create_node() : error, call to cudaMallocHost() has failed for host_Weights_Transposed for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*insize*outsize, outsize, insize, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMalloc((void **) &new_node->device_Weights_Transposed, sizeof(float)*insize*outsize)) != cudaSuccess ){ // weights are insize x outsize
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for device_Weights_Transposed for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*insize*outsize, outsize, insize, cudaGetErrorString(cudret));
        return NULL;
    }
// reserve memory for temporary derivative calculation
    if( (cudret=cudaMallocHost((void **) &new_node->host_Delta, sizeof(float)*outsize)) != cudaSuccess ){ // weights are insize x outsize
        fprintf(stderr, "create_node() : error, call to cudaMallocHost() has failed for host_Delta for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }
    if( (cudret=cudaMalloc((void **) &new_node->device_Delta, sizeof(float)*outsize)) != cudaSuccess ){ // weights are insize x outsize
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for device_Delta for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, outsize, 1, cudaGetErrorString(cudret));
        return NULL;
    }

    new_node->next = NULL; 
    new_node->prev = NULL; 
    return new_node; 
} 

int add_node(int insize, int outsize, SV *biases, SV *weights)
{
    node_t* new_node = create_node(insize, outsize, biases, weights); 
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

int reserve_input_memory(int insize, int outsize)
{
    cudaError_t cudret;
// memory for input array
    if( (cudret=cudaMallocHost((void **) &host_x, sizeof(float)*insize)) != cudaSuccess ){ // initial input is 1 x insize
        fprintf(stderr, "reserve_input_memory() : error, call to cudaMallocHost() has failed for host_x for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*1*insize, insize, 1, cudaGetErrorString(cudret));
        return 0;
    }
    if( (cudret=cudaMalloc((void **) &device_x, sizeof(float)*insize)) != cudaSuccess ){   // initial_input is 1 x insize
        fprintf(stderr, "reserve_input_memory() : error, call to cudaMalloc() has failed for device_x for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*insize, insize, 1, cudaGetErrorString(cudret));
        return 0;
    }
    if( (cudret=cudaMalloc((void **) &device_x_transposed, sizeof(float)*insize)) != cudaSuccess ){   // initial_input is 1 x insize
        fprintf(stderr, "reserve_input_memory() : error, call to cudaMalloc() has failed for device_x_transposed for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*insize, 1, insize, cudaGetErrorString(cudret));
        return 0;
    }
// memory for target
    if( (cudret=cudaMallocHost((void **) &host_y, sizeof(float)*outsize)) != cudaSuccess ){ // target is 1 x outsize
        fprintf(stderr, "reserve_input_memory() : error, call to cudaMallocHost() has failed for host_y for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*1*outsize, outsize, 1, cudaGetErrorString(cudret));
        return 0;
    }
    if( (cudret=cudaMalloc((void **) &device_y, sizeof(float)*outsize)) != cudaSuccess ){  // target is 1 x outsize
        fprintf(stderr, "reserve_input_memory() : error, call to cudaMalloc() has failed for device_y for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, outsize, 1, cudaGetErrorString(cudret));
        return 0;
    }
// memory for cost derivative, 1 x outsize
    if( (cudret=cudaMallocHost((void **) &host_Cost_Derivative, sizeof(float)*outsize)) != cudaSuccess ){
        fprintf(stderr, "create_node() : error, call to cudaMallocHost() has failed for host_Cost_Derivative for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, outsize, 1, cudaGetErrorString(cudret));
        return 0;
    }   
    if( (cudret=cudaMalloc((void **) &device_Cost_Derivative, sizeof(float)*outsize)) != cudaSuccess ){
        fprintf(stderr, "create_node() : error, call to cudaMalloc() has failed for device_Cost_Derivative for %zu bytes (matrix size: (h/rows,w/cols)=(%zu,%zu): %s\n", sizeof(float)*outsize, outsize, 1, cudaGetErrorString(cudret)); 
        return 0;
    }

    return 1;
}


int get_last_activated_output( SV *R ) {

    // Transfer results from host to perl
    AV *av, *av2;
    float *pd;
    size_t i,j,RH,RW, asz;
    cudaError_t cudret;

    RW = 1;
    RH = tail->output_size;

// copy device data back to the host before loading the Perl values

    if( (cudret=cudaMemcpy(tail->host_Activated_Output, tail->device_Activated_Output, RW*RH*sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess ){
        fprintf(stderr, "inline_cuda_matrix_sigmoid_prime() : error, call to cudaMemcpy(cudaMemcpyDeviceToHost) has failed for transfering results to host device: %s\n", cudaGetErrorString(cudret));
        return 1;
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

void run_update_weights_and_biases(float modifier) {
   cudaError_t cudret;
   node_t * current = head;

   while (current != NULL) {
      run_gpu_update_weights(modifier, current); 
      run_gpu_update_biases(modifier, current);
      current = current->next;
   }
}

EOCODE
}

use Inline CUDA => Config =>
        host_code_language => 'c',
        BUILD_NOISY => 1,
        clean_after_build => 0,
        warnings => 10,
;

use Inline CUDA => $code;

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

sub print_network {
   print_list();
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

sub calculate_loss {
   my $R = [];
   calculate_cost_and_derivative($R) and die;
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

sub show_final_weights {
  run_show_final_weights();
}

1;
