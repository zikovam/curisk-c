#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <time.h>
#include "curisk.cuh"

__global__ void generate_vector_sample_kernel();
__global__ void setup_gamma_generator(long seed);
__device__ __forceinline__ float generate_gamma_1_1(curandState *state);

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("\nError \"%s\" at %s:%d\n", cudaGetErrorString(x), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);}} while(0)

#define check_error() do { if(cudaGetLastError()!=cudaSuccess) { \
    printf("\nError \"%s\" at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);}} while(0)

#define ESTIMATED_MAX_DIMENSION 32

__constant__ int c_dimension;
__constant__ int c_vector_scheme[ESTIMATED_MAX_DIMENSION];
__constant__ bound_t c_vector_bounds[ESTIMATED_MAX_DIMENSION];
__constant__ ordinal_t c_vector_ordinal[ESTIMATED_MAX_DIMENSION];
__constant__ int c_sample_size;

__device__ float *d_vector_sample;
__device__ int d_round_sample_size;
__device__ float *d_round_vector_sample;
__device__ curandState *d_curand_states;
__device__ int d_vectors_ready;

void generate_vector_sample(sampling_cofiguration_t& conf, sampling_result_t& result, int timeout_rounds)
{
    int start_time_point = clock();

    /* Скопируем некоторые поля из conf в константную память. */
    conf.log() << "Preparing constant variables." << std::endl;
    cudaMemcpyToSymbol(c_dimension, &conf.dimension, sizeof(int)); check_error();
    cudaMemcpyToSymbol(c_sample_size, &conf.sample_size, sizeof(int)); check_error();
    cudaMemcpyToSymbol(c_vector_scheme, conf.vector_scheme, conf.dimension*sizeof(int)); check_error();
    cudaMemcpyToSymbol(c_vector_bounds, conf.vector_bounds, conf.dimension*sizeof(bound_t)); check_error();
    cudaMemcpyToSymbol(c_vector_ordinal, conf.vector_ordinal, conf.dimension*sizeof(ordinal_t)); check_error();

    /* Выделим память для выборки. */
    conf.log() << "Allocate memory for vector sample." << std::endl;
    float *dh_vector_sample;
    cudaMalloc(&dh_vector_sample, conf.sample_size*conf.dimension*sizeof(float)); check_error();
    cudaMemcpyToSymbol(d_vector_sample, &dh_vector_sample, sizeof(dh_vector_sample)); check_error();

    /* Выделим память для выборки раунда. */
    conf.log() << "Allocate memory for round vector sample." << std::endl;
    int blocks_per_round = conf.grid_dimension.x;
    int vectors_per_block = conf.block_dimension.x;
    int round_sample_size = blocks_per_round*vectors_per_block;
    cudaMemcpyToSymbol(d_round_sample_size, &round_sample_size, sizeof(int)); check_error();

    float *dh_round_vector_sample;
    cudaMalloc(&dh_round_vector_sample, round_sample_size*conf.dimension*sizeof(float)); check_error();
    cudaMemcpyToSymbol(d_round_vector_sample, &dh_round_vector_sample, sizeof(dh_round_vector_sample)); check_error();

    /* Настроим генератор случайных чисел. */
    conf.log() << "Setup CUDA random numbers generator." << std::endl;
    curandState *dh_curand_states;
    cudaMalloc(&dh_curand_states, round_sample_size*sizeof(curandState)); check_error();
    cudaMemcpyToSymbol(d_curand_states, &dh_curand_states, sizeof(dh_curand_states)); check_error();

    setup_gamma_generator<<<blocks_per_round, vectors_per_block>>>(clock()); check_error();
    cudaDeviceSynchronize(); check_error();

    /* Число готовых элементов выборки. */
    int vectors_ready = 0;
    cudaMemcpyToSymbol(d_vectors_ready, &vectors_ready, sizeof(vectors_ready)); check_error();

    conf.log() << "Start round cycle." << std::endl;
    int rounds = 0;
    while (vectors_ready < conf.sample_size)
    {
        generate_vector_sample_kernel<<<blocks_per_round, vectors_per_block>>>();
        CUDA_CALL(cudaDeviceSynchronize());
        cudaMemcpyFromSymbol(&vectors_ready, d_vectors_ready, sizeof(vectors_ready)); check_error();
        rounds++;
        if (rounds > timeout_rounds)
        {
            conf.log() << "Round cycle is terminated (timeout)." << std::endl;
            result.generated_vectors_number = vectors_ready;
            result.error = SAMPLING_TIMEOUT;
            break;
        }
    }

    conf.log() << "Stop round cycle." << std::endl;
    conf.log() << "Vectors generated: " << vectors_ready << "/" << conf.sample_size << "." << std::endl;

    if (vectors_ready < conf.sample_size)
    {
        cudaMemcpy(result.vector_sample, dh_vector_sample, vectors_ready*conf.dimension*sizeof(float), cudaMemcpyDeviceToHost); check_error();
    }
    else
    {
        cudaMemcpy(result.vector_sample, dh_vector_sample, conf.sample_size*conf.dimension*sizeof(float), cudaMemcpyDeviceToHost); check_error();
    }
    
    cudaFree(dh_vector_sample);
    cudaFree(dh_round_vector_sample);
    cudaFree(dh_curand_states);

    int end_time_point = clock();
    float elapsed_time = ((float) (end_time_point - start_time_point))/CLOCKS_PER_SEC;
    conf.log() << "Elapsed time: " << elapsed_time << " s." << std::endl;
    result.elapsed_time = elapsed_time;
}

__global__
void generate_vector_sample_kernel()
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int dimension = c_dimension;
    int i;
    float value;
    float than_value;
    float sum = 0;
    int last_vectors_ready;
    ordinal_t ordinal;
    bound_t bound;
    bool eliminate = false;

    for (i = 0; i < dimension; i++)
    {
        if (c_vector_scheme[i] == 0)
            value = 0;
        else
            value = generate_gamma_1_1(d_curand_states + idx);

        sum += value;
        d_round_vector_sample[dimension*idx + i] = value;
    }

    if (sum != 0)
    {
        for (i = 0; i < dimension; i++)
        {
            value = d_round_vector_sample[dimension*idx + i];
            value /= sum;
            d_round_vector_sample[dimension*idx + i] = value;
            bound = c_vector_bounds[i];
            eliminate = eliminate || value < bound.left || value > bound.right;
        }

        for (i = 0; i < dimension; i++)
        {
            value = d_round_vector_sample[dimension*idx + i];
            ordinal = c_vector_ordinal[i];
            than_value = d_round_vector_sample[dimension*idx + ordinal.than_index];
            eliminate = eliminate ||
                (ordinal.ordinal == ORDINAL_LESS && value >= than_value) ||
                (ordinal.ordinal == ORDINAL_MORE && value <= than_value);
        }
    }

    if (!eliminate)
    {
        last_vectors_ready = atomicAdd(&d_vectors_ready, 1);
        if (last_vectors_ready < c_sample_size)
        {
            for (i = 0; i < dimension; i++)
                d_vector_sample[dimension*last_vectors_ready + i] = d_round_vector_sample[dimension*idx + i];
        }
    }
}

__global__
void setup_gamma_generator(long seed)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < d_round_sample_size)
        curand_init(seed, tid, 0, &d_curand_states[tid]);
}

/*
 * Функция генерирует случайную величину с распределением Gamma(1,1).
 */
__device__ __forceinline__
float generate_gamma_1_1(curandState *state)
{
    curandState localState = *state;
    float c, z, u, v, result;

    c = 1/sqrtf(9*2/3.);

    do {
        z = curand_normal(&localState);
        u = curand_uniform(&localState);
        v = powf(1 + c*z, 3);
    } while (z <= (-1/c) || logf(u) >= (0.5*z*z + 2/3. - (2/3.)*v + (2/3.)*logf(v)));

    result = (2/3.)*v;

    *state = localState;

    return result;
}