
// CUDA implementation of applying 3x3 low-pass filter to a JPEG image
//

#include <iostream>
#include <cuda_runtime.h> // CUDA Header
#include "utils.hpp"

const int FILTER_SIZE = 3;
const float filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

__constant__ float constantFilter[FILTER_SIZE*FILTER_SIZE];

// CUDA kernel function: 3x3 low-pass filter
__global__ void ApplyFilter(const unsigned char* input, unsigned char* output,
                              int width, int height, int num_channels)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > 0 && row < height - 1 && col > 0 && col < width - 1){
        unsigned char sum[3] = {0,0,0};

        unsigned char upper[9];
        unsigned char middle[9];
        unsigned char lower[9];

        memcpy(&upper, &input[((row - 1) * width + col - 1) * num_channels], 9 * sizeof(unsigned char));
        memcpy(&middle, &input[(row * width + col  - 1) * num_channels], 9 * sizeof(unsigned char));
        memcpy(&lower, &input[((row + 1) * width + col  - 1) * num_channels], 9 * sizeof(unsigned char));

        sum[0] = upper[0]*constantFilter[0] + upper[3]*constantFilter[1] + upper[6]*constantFilter[2] + 
            middle[0]*constantFilter[3] + middle[3]*constantFilter[4] + middle[6]*constantFilter[5] +
            lower[0]*constantFilter[6] + lower[3]*constantFilter[7] + lower[6]*constantFilter[8];

        sum[1] = upper[1]*constantFilter[0] + upper[4]*constantFilter[1] + upper[7]*constantFilter[2] + 
            middle[1]*constantFilter[3] + middle[4]*constantFilter[4] + middle[7]*constantFilter[5] +
            lower[1]*constantFilter[6] + lower[4]*constantFilter[7] + lower[7]*constantFilter[8];

        sum[2] = upper[2]*constantFilter[0] + upper[5]*constantFilter[1] + upper[8]*constantFilter[2] +
            middle[2]*constantFilter[3] + middle[5]*constantFilter[4] + middle[8]*constantFilter[5] +
            lower[2]*constantFilter[6] + lower[5]*constantFilter[7] + lower[8]*constantFilter[8];



        memcpy(&output[(row * width + col) * num_channels],&sum,3 * sizeof(unsigned char));

    }

}

int main(int argc, char** argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    // Allocate memory on host (CPU)
    unsigned char* outputImage = new unsigned char[input_jpeg.width * input_jpeg.height *
                                                   input_jpeg.num_channels];
    // Allocate memory on device (GPU)
    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc((void**)&d_input, input_jpeg.width * input_jpeg.height *
                                     input_jpeg.num_channels *
                                     sizeof(unsigned char));
    cudaMalloc((void**)&d_output, input_jpeg.width * input_jpeg.height *
                                      input_jpeg.num_channels *
                                      sizeof(unsigned char));

    float flattened_filter[FILTER_SIZE * FILTER_SIZE];
    for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            flattened_filter[i * FILTER_SIZE + j] = filter[i][j];
        }
    }

    cudaMemcpyToSymbol(constantFilter, flattened_filter, FILTER_SIZE * FILTER_SIZE * sizeof(float));
    // Copy input data from host to device
    cudaMemcpy(d_input, input_jpeg.buffer,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels *
                   sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    // Computation: 3x3 Low-pass filter
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 blockSize(32, 32);
    dim3 numBlocks((input_jpeg.width + blockSize.x - 1) / blockSize.x,
                   (input_jpeg.height + blockSize.y - 1) / blockSize.y);
    cudaEventRecord(start, 0); // GPU start time
    ApplyFilter<<<numBlocks, blockSize>>>(d_input, d_output, input_jpeg.width,
                                            input_jpeg.height, input_jpeg.num_channels);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from device to host
    cudaMemcpy(outputImage, d_output,
               input_jpeg.width * input_jpeg.height * input_jpeg.num_channels *
                   sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    // Write output image to JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{outputImage, input_jpeg.width, input_jpeg.height,
                         input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory on device and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] input_jpeg.buffer;
    delete[] outputImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
