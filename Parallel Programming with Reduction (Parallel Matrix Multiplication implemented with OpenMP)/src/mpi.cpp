// 120090712 
//MPI + OpenMp + SIMD + Reordering Matrix Multiplication


#include <mpi.h>  // MPI Header
#include <omp.h> 
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

#define MASTER 0

Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, size_t start_row, size_t end_row) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    #pragma omp parallel for
    for (size_t i = start_row; i < end_row; ++i) {
        int * result_row = result[i];
        const int* left_row = matrix1[i];
        for (size_t k = 0; k < K; ++k) {
            const int* a = left_row + k;
            const int* right_row = matrix2[k];

            __m256i ascalar = _mm256_set1_epi32(static_cast<float>(*a));

            for (size_t j = 0; j < N; j += 8) {
                const int* b = right_row + j;

                // Boudary Check
                size_t remaining = N - j;
                if (remaining >= 8) {
                    __m256i eight_intger_b = _mm256_loadu_si256((__m256i*)b);
                    __m256i multi_result = _mm256_mullo_epi32(eight_intger_b, ascalar);
                    __m256i previous_value = _mm256_loadu_si256((__m256i*)(result_row + j));
                    __m256i final_result = _mm256_add_epi32(multi_result, previous_value);
                    _mm256_storeu_si256((__m256i*)(result_row + j), final_result);
                } else {
                    // One by one
                    for (size_t t = 0; t < remaining; ++t) {
                        *(result_row + j + t) += (*a) * b[t];
                    }
                }
            }
        }

    }
    return result;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    size_t cuts[numtasks + 1] = {0};
    size_t total_rows = matrix1.getRows();
    size_t rows_per_task = total_rows/numtasks;
    size_t left_rows_num = total_rows%numtasks;
    size_t divided_left_rows_num = 0;

    for (int i = 0; i < numtasks; i++) {
        if (divided_left_rows_num < left_rows_num) {
            cuts[i+1] = cuts[i] + rows_per_task + 1;
            divided_left_rows_num++;
        } else cuts[i+1] = cuts[i] + rows_per_task;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (taskid == MASTER) {

        // Matrix Final_Matrix(total_rows,matrix2.getCols()); 
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, cuts[taskid], cuts[taskid + 1]);      
        MPI_Request request[total_rows-cuts[taskid + 1]];
        for (size_t i = cuts[taskid + 1]; i < total_rows; ++i){
            MPI_Irecv(result[i], matrix2.getCols(), MPI_INT, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, &request[i - cuts[taskid + 1]]);
        }
        MPI_Waitall(total_rows - cuts[taskid + 1], &request[0], MPI_STATUS_IGNORE);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;

        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds" << std::endl;
    } else {
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, cuts[taskid], cuts[taskid + 1]);
        MPI_Request request[cuts[taskid + 1]- cuts[taskid]];
        for (size_t i = cuts[taskid]; i < cuts[taskid + 1]; ++i){
            MPI_Isend(result[i], matrix2.getCols(), MPI_INT, MASTER, i, MPI_COMM_WORLD, &request[i-cuts[taskid]]);
        }
        MPI_Waitall(cuts[taskid + 1]- cuts[taskid], &request[0], MPI_STATUS_IGNORE);
    }

    MPI_Finalize();
    return 0;
}
