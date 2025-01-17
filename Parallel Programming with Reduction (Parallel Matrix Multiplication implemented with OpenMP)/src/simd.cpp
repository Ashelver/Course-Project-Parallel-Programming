// 120090712
// SIMD + Reordering Matrix Multiplication

#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

Matrix matrix_multiply_simd(const Matrix& matrix1, const Matrix& matrix2) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    for (size_t i = 0; i < M; ++i) {
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
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_simd(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}