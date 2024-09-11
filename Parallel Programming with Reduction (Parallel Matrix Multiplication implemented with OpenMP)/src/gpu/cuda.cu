// 120090712
// CUDA Matrix Multiplication

#include <chrono>
#include <iostream>
#include "../matrix.hpp"

__global__ void matrix_multiply_kernel(const int* matrix1, const int* matrix2, int* result, size_t M, size_t K, size_t N) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        int sum = 0;
        for (size_t k = 0; k < K; ++k) {
            sum += matrix1[i * K + k] * matrix2[k * N + j];
        }
        result[i * N + j] = sum;
    }
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr <<
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n";
        return - 1;
    }

    const std::string matrix1_path = argv[1];
    const std::string matrix2_path = argv[2];
    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    // Start Calculation!
    if (matrix1.getCols() != matrix2.getRows()) {
        std::cerr <<
            "Matrix dimensions are not compatible for multiplication.";
        exit(-1);
    }
    
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    int* d_matrix1;
    int* d_matrix2;
    int* d_result;

    int* flatten_matrix1 = new int[M * K];
    int* flatten_matrix2 = new int[K * N];
    int* flatten_result = new int[M * N];

    for (int i = 0; i < M; ++i){
        memcpy(flatten_matrix1 + i * K,matrix1[i], K* sizeof(int));
    }

    for (int i = 0; i < K; ++i){
        memcpy(flatten_matrix2 + i * N,matrix2[i], N* sizeof(int));
    }

    size_t matrix1_size = M * K * sizeof(int);
    size_t matrix2_size = K * N * sizeof(int);
    size_t result_size = M * N * sizeof(int);



    cudaMalloc((void**)&d_matrix1, matrix1_size);
    auto start_time = std::chrono::high_resolution_clock::now();


    cudaMalloc((void**)&d_matrix2, matrix2_size);
    cudaMalloc((void**)&d_result, result_size);


    cudaMemcpy(d_matrix1, flatten_matrix1, matrix1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, flatten_matrix2, matrix2_size, cudaMemcpyHostToDevice);


    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);

    matrix_multiply_kernel<<<grid_size, block_size>>>(d_matrix1, d_matrix2, d_result, M, K, N);

    cudaMemcpy(flatten_result, d_result, result_size, cudaMemcpyDeviceToHost);

    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);

    for (int i = 0; i < M; ++i){
        memcpy(result[i], flatten_result + i * N, N* sizeof(int));
    }
    delete[] flatten_matrix1;
    delete[] flatten_matrix2;
    delete[] flatten_result;

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



