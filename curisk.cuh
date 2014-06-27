#ifndef CURISK_H
#define CURISK_H

#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>

/*
 * ��������� ���������� ������� �������� ���������� �������.
 */
struct bound_t
{
    float left;
    float right;
};

#define BOUND_INIT {0, 1}

enum ordinal_e
{
    ORDINAL_LESS,
    ORDINAL_MORE,
    ORDINAL_NEVER_MIND
};

/*
 * ����������� ����������� �������. � ���� than_index ����������� ������ ��������,
 * �������������� �������� ������ ������� (� �������� ��������� ���������) ������ ���
 * ������.
 * 
 * ���� �� ������� �� ������������� ����������� �����������, �� than_index == 0 �
 * ordinal == ORDINAL_NEVER_MIND.
 */
struct ordinal_t
{
    int than_index;
    ordinal_e ordinal;
};

#define ORDINAL_INIT {0, ORDINAL_NEVER_MIND}

/*
 * ������������ ���������� �������, ������� ���������� ��� �� ��� �������. � ���
 * ���������� ��������� ������ � ������������ ���� CUDA (����������� ����� � �����).
 */
struct sampling_cofiguration_t
{
    sampling_cofiguration_t(int dimension, int sample_size, dim3 grid_dimension,
        dim3 block_dimension, std::ostream& log = std::cout)
    {
        this->dimension = dimension;
        this->sample_size = sample_size;
        this->grid_dimension = grid_dimension;
        this->block_dimension = block_dimension;
        this->_log = &log;
        vector_scheme = new int[dimension];
        vector_bounds = new bound_t[dimension];
        vector_ordinal = new ordinal_t[dimension];
    }

    ~sampling_cofiguration_t()
    {
        delete [] vector_scheme;
        delete [] vector_bounds;
        delete [] vector_ordinal;
    }

    int dimension; // ������ ������� W.
    int *vector_scheme; // ����� ������� W ���� (0, 1, 1, 0 ...).
    bound_t *vector_bounds;
    ordinal_t *vector_ordinal;
    int sample_size; // ���������� �������� � �������� ������� ��������.
    dim3 grid_dimension;
    dim3 block_dimension;
    std::ostream& log() { return *_log; }

private:
    std::ostream *_log;
};

/*
 * ��� ������, ������������ �������� ��������� ��������
 * � ��������� sampling_result_t.
 */
enum sampling_error_code_e
{
    SAMPLING_SUCCESS,
    SAMPLING_TIMEOUT,
    SAMPLING_CUDA_ERROR,
    SAMPLING_UNKNOWN_ERROR
};

/*
 * ��������� ���������� ��������� ��������. ����������� ��������
 * ��������� ��������.
 *
 * � ���� generated_vectors_number �������� ����� ��������������� ��������
 * �������. ���� ��� ��������� ���������, ��� ����� ���� ������� ��
 * ����, ������� ������� �� ����� �� ����.
 */
struct sampling_result_t
{
    sampling_result_t(sampling_cofiguration_t *conf)
    {
        size_t size = conf->dimension*conf->sample_size;
        vector_sample = new float[size];
        memset(vector_sample, 0, size*sizeof(float));
        error = SAMPLING_SUCCESS;
        cuda_error = cudaSuccess;
        generated_vectors_number = 0;
    }

    ~sampling_result_t()
    {
        delete [] vector_sample;
    }

    float elapsed_time;
    sampling_error_code_e error;
    cudaError_t cuda_error;
    int generated_vectors_number;
    float *vector_sample;
};

/*
 * ������� ��������� ��������. �������� timeout_rounds ��������� �������� ����� ������� (�����
 * �������� ���� CUDA), ����� �������� ��������� ����������� � �������. ������ ����������� �
 * ��������� result ��� SAMPLING_TIMEOUT.
 */
void generate_vector_sample(sampling_cofiguration_t& conf, sampling_result_t& result, int timeout_rounds);

#endif