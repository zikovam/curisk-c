#ifndef CURISK_H
#define CURISK_H

#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>

/*
 * Структура определяет границы элемента случайного вектора.
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
 * Ординальное ограничение вектора. В поле than_index указывается индекс элемента,
 * относительного которого данный элемент (к которому относится структура) больше или
 * меньше.
 * 
 * Если на элемент не накладываются ординальные ограничения, то than_index == 0 и
 * ordinal == ORDINAL_NEVER_MIND.
 */
struct ordinal_t
{
    int than_index;
    ordinal_e ordinal;
};

#define ORDINAL_INIT {0, ORDINAL_NEVER_MIND}

/*
 * Конфигурация генератора выборки, которая передается ему до его запуска. В ней
 * содержатся параметры отбора и конфигурация ядра CUDA (размерность сетки и блока).
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

    int dimension; // Размер веткора W.
    int *vector_scheme; // Схема вектора W вида (0, 1, 1, 0 ...).
    bound_t *vector_bounds;
    ordinal_t *vector_ordinal;
    int sample_size; // Количество векторов в желаемой выборке векторов.
    dim3 grid_dimension;
    dim3 block_dimension;
    std::ostream& log() { return *_log; }

private:
    std::ostream *_log;
};

/*
 * Тип ошибки, возвращаемый функцией генерации векторов
 * в структуре sampling_result_t.
 */
enum sampling_error_code_e
{
    SAMPLING_SUCCESS,
    SAMPLING_TIMEOUT,
    SAMPLING_CUDA_ERROR,
    SAMPLING_UNKNOWN_ERROR
};

/*
 * Структура результата генерации векторов. Заполняется функцией
 * генерации векторов.
 *
 * В поле generated_vectors_number хранится число сгенерированных векторов
 * выборки. Даже при неудачной генерации, оно может быть отлично от
 * нуля, поэтому неплохо бы знать об этом.
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
 * Функция генерации векторов. Значение timeout_rounds указывает порогове число раундов (число
 * запусков ядра CUDA), после которого генерация завершается с ошибкой. Ошибка фиксируется в
 * структуре result как SAMPLING_TIMEOUT.
 */
void generate_vector_sample(sampling_cofiguration_t& conf, sampling_result_t& result, int timeout_rounds);

#endif