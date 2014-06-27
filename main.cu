#include <iostream>
#include <iomanip>
#include <fstream>

#include "curisk.cuh"
#include "nullbuf.cuh"

//std::ofstream logFile("log.txt", std::ofstream::out);
std::ostream& logger = std::cout;

#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 64

void write_sample(std::ostream& output, sampling_cofiguration_t& conf, sampling_result_t& result, std::streamsize precision = 4)
{
    std::streamsize width = precision + 2;
    for (int i = 0; i < conf.sample_size; i++)
    {
        for (int j = 0; j < conf.dimension; j++)
        {
            float value = result.vector_sample[conf.dimension*i + j];
            output << std::fixed << std::setw(width) << std::setprecision(precision) 
                << std::setfill('0') << value << " ";
        }
        output << std::endl;
    }
}

void curisk_test_simple()
{
    logger << "Running curisk_test_simple." << std::endl;

    int vector_scheme[3] = {1, 1, 1};
    bound_t vector_bounds[3] = {BOUND_INIT, BOUND_INIT, BOUND_INIT};
    ordinal_t vector_ordinal[3] = {ORDINAL_INIT, ORDINAL_INIT, ORDINAL_INIT};

    sampling_cofiguration_t conf(3, 1000, dim3(BLOCKS_PER_GRID, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1));
    memcpy(conf.vector_scheme, vector_scheme, sizeof(vector_scheme));
    memcpy(conf.vector_bounds, vector_bounds, sizeof(vector_bounds));
    memcpy(conf.vector_ordinal, vector_ordinal, sizeof(vector_ordinal));
    sampling_result_t result(&conf);
    generate_vector_sample(conf, result, 5000);

    std::ofstream output("curisk_test_simple.txt", std::ofstream::out);
    write_sample(output, conf, result);
    output.close();

    logger << std::endl;
}

void curisk_test_bounds()
{
    logger << "Running curisk_test_bounds." << std::endl;

    int vector_scheme[3] = {1, 1, 1};
    bound_t vector_bounds[3] = {{0.2, 0.7}, {0.1, 0.8}, {0.5, 0.9}};
    ordinal_t vector_ordinal[3] = {ORDINAL_INIT, ORDINAL_INIT, ORDINAL_INIT};

    sampling_cofiguration_t conf(3, 1000, dim3(BLOCKS_PER_GRID, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1));
    memcpy(conf.vector_scheme, vector_scheme, sizeof(vector_scheme));
    memcpy(conf.vector_bounds, vector_bounds, sizeof(vector_bounds));
    memcpy(conf.vector_ordinal, vector_ordinal, sizeof(vector_ordinal));
    sampling_result_t result(&conf);
    generate_vector_sample(conf, result, 5000);

    std::ofstream output("curisk_test_bounds.txt", std::ofstream::out);
    write_sample(output, conf, result);
    output.close();

    logger << std::endl;
}

void curisk_test_ordinal()
{
    logger << "Running curisk_test_ordinal." << std::endl;

    int vector_scheme[3] = {1, 1, 1};
    bound_t vector_bounds[3] = {BOUND_INIT, BOUND_INIT, BOUND_INIT};
    ordinal_t vector_ordinal[3] = {
        {2, ORDINAL_MORE},
        {2, ORDINAL_LESS},
        ORDINAL_INIT
    };

    sampling_cofiguration_t conf(3, 1000, dim3(BLOCKS_PER_GRID, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1));
    memcpy(conf.vector_scheme, vector_scheme, sizeof(vector_scheme));
    memcpy(conf.vector_bounds, vector_bounds, sizeof(vector_bounds));
    memcpy(conf.vector_ordinal, vector_ordinal, sizeof(vector_ordinal));
    sampling_result_t result(&conf);
    generate_vector_sample(conf, result, 5000);

    std::ofstream output("curisk_test_ordinal.txt", std::ofstream::out);
    write_sample(output, conf, result);
    output.close();

    logger << std::endl;
}

void curisk_test_ordinal_and_bounds()
{
    logger << "Running curisk_test_ordinal_and_bounds." << std::endl;

    int vector_scheme[3] = {1, 1, 1};
    bound_t vector_bounds[3] = {{0.2, 0.3}, {0.1, 0.8}, {0.5, 0.9}};
    ordinal_t vector_ordinal[3] = {
        {1, ORDINAL_MORE},
        {2, ORDINAL_LESS},
        ORDINAL_INIT
    };

    sampling_cofiguration_t conf(3, 1000, dim3(BLOCKS_PER_GRID, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1));
    memcpy(conf.vector_scheme, vector_scheme, sizeof(vector_scheme));
    memcpy(conf.vector_bounds, vector_bounds, sizeof(vector_bounds));
    memcpy(conf.vector_ordinal, vector_ordinal, sizeof(vector_ordinal));
    sampling_result_t result(&conf);
    generate_vector_sample(conf, result, 5000);

    std::ofstream output("curisk_test_ordinal_and_bounds.txt", std::ofstream::out);
    write_sample(output, conf, result);
    output.close();

    logger << std::endl;
}

void curisk_test_bounds_increase()
{
    // Будем увеличивать левую границу второго элемента.
    logger << "Running curisk_test_bounds_increase." << std::endl;

    int vector_scheme[3] = {1, 1, 1};
    bound_t vector_bounds[3] = {{0.2, 0.3}, {0.1, 0.8}, {0.5, 0.9}};
    ordinal_t vector_ordinal[3] = {ORDINAL_INIT, ORDINAL_INIT, ORDINAL_INIT};

    sampling_cofiguration_t conf(3, 1000, dim3(BLOCKS_PER_GRID, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1));
    memcpy(conf.vector_scheme, vector_scheme, sizeof(vector_scheme));
    memcpy(conf.vector_bounds, vector_bounds, sizeof(vector_bounds));
    memcpy(conf.vector_ordinal, vector_ordinal, sizeof(vector_ordinal));
    sampling_result_t result(&conf);

    float left_bound = 0.1;
    float step = 0.001;
    int times = 10;
    std::ofstream output("curisk_test_bounds_increase.txt", std::ofstream::out);

    do
    {
        float total_time = 0;
        float average_time = 0;
        for (int i = 0; i < times; i++)
        {
            conf.vector_bounds[1].left = left_bound;
            generate_vector_sample(conf, result, 5000);
            if (result.error == SAMPLING_TIMEOUT)
                break;
            total_time += result.elapsed_time;
        }
        average_time = total_time/times;
        output << left_bound << " " << average_time << std::endl;
        left_bound += step;
    }
    while (result.error != SAMPLING_TIMEOUT);

    output.close();

    logger << std::endl;
}

void curisk_test_time()
{
    logger << "Running curisk_test_time." << std::endl;

    int vector_scheme[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    bound_t vector_bounds[10] = {
        BOUND_INIT, BOUND_INIT, BOUND_INIT, BOUND_INIT, BOUND_INIT,
        BOUND_INIT, BOUND_INIT, BOUND_INIT, BOUND_INIT, BOUND_INIT,
    };
    ordinal_t vector_ordinal[10] = {
        ORDINAL_INIT, ORDINAL_INIT, ORDINAL_INIT, ORDINAL_INIT, ORDINAL_INIT,
        ORDINAL_INIT, ORDINAL_INIT, ORDINAL_INIT, ORDINAL_INIT, ORDINAL_INIT
    };

    int n_vectors = 1000000;

    sampling_cofiguration_t conf(10, n_vectors, dim3(BLOCKS_PER_GRID, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1), cnull);
    memcpy(conf.vector_scheme, vector_scheme, sizeof(vector_scheme));
    memcpy(conf.vector_bounds, vector_bounds, sizeof(vector_bounds));
    memcpy(conf.vector_ordinal, vector_ordinal, sizeof(vector_ordinal));
    sampling_result_t result(&conf);

    int times = 30;
    float total = 0;
    float average = 0;
    for (int i = 0; i < times; i++)
    {
        generate_vector_sample(conf, result, 5000);
        total += result.elapsed_time;
    }
    average = total/times;

    logger << "Dimension: " << 10 << "." << std::endl;
    logger << "Generated vectors: " << n_vectors << "." << std::endl; 
    logger << "Average time: " << average << " s." << std::endl; 
    logger << std::endl;
}

int main(int argc, char *argv[])
{
    /*
     * При первом запуске ядер CUDA производится их компиляция в Runtime.
     * Это занимает сравнителньо много времени, поэтому запустим один тест
     * вхолостую.
     */
    curisk_test_simple();

    curisk_test_simple();
    curisk_test_bounds();
    curisk_test_ordinal();
    curisk_test_ordinal_and_bounds();
    //curisk_test_bounds_increase();
    curisk_test_time();

    cudaDeviceReset();

    return EXIT_SUCCESS;
}