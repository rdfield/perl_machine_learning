int run_gpu_linear( float *activation, float *device_Weights,  float *device_Bias, float *device_Output, int m, int n, int k) ;
int run_gpu_matmul( float *a, float *b,  float *r, int input_size, int middle_size, int output_size) ;
int run_gpu_derivative( float *a, float *b,  float *sp, float *r, int input_size, int middle_size, int output_size) ;
int run_gpu_weight_derivative( float *a, float *b, float *r, int input_size, int middle_size, int output_size);
int gpu_sigmoid( float *device_Output, float *device_Activated_Output, int output_size , int cols);
int run_gpu_sigmoid_prime( float *device_Activated_Output, float *device_Activated_Output_Derivative, int output_size, int cols );
int gpu_add_same_size( float *lhs, float *rhs, int arraysize );
void run_gpu_transpose_2D_array( float *in, float *transpose, size_t rows, size_t cols) ;
void run_gpu_update_weights( float modifier, float *device_Weights, float *device_Weights_Derivative, int output_size, int input_size) ;
void run_gpu_update_biases( float modifier, float *device_Bias, float *device_Bias_Derivative, int output_size, int batch_size);
void gpu_calculate_cost_and_derivative(float *activated_output, float *y, float *cost_derivative, size_t rows, size_t cols) ;



void gpu_memcpy_to_device( float *host_data, float *device_data, size_t size_data);
void gpu_memcpy_intra_device( float *from_data, float *to_data, size_t size_data);
void gpu_memcpy_from_device( float *host_data, float *device_data, size_t size_data);


float * gpu_device_malloc(  size_t size_data);
float * gpu_host_malloc(  size_t size_data);
