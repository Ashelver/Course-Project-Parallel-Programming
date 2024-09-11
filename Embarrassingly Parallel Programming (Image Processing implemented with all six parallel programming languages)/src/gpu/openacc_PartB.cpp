// OpenACC implementation of applying a 3x3 low-pass filter to a JPEG image
//

#include <iostream>
#include <chrono>

#include "utils.hpp"
// #include <openacc.h> // OpenACC Header


const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};


int main(int argc, char **argv)
{
    // Verify input argument format
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read from input JPEG
    const char *input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    JPEGMeta input_jpeg = read_from_jpeg(input_filepath);
    // Computation: Apply 3*3 filter
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    unsigned char *filteredImage = new unsigned char[width * height * num_channels];
    for (int i = 0; i < width * height * num_channels; i++){
        filteredImage[i] = 0;
    }

    float flattened_filter[FILTER_SIZE * FILTER_SIZE];
    for (int i = 0; i < FILTER_SIZE; i++) {
        for (int j = 0; j < FILTER_SIZE; j++) {
            flattened_filter[i * FILTER_SIZE + j] = filter[i][j];
        }
    }

    unsigned char *buffer = new unsigned char[width * height * num_channels];
    for (int i = 0; i < width * height * num_channels; i++)
    {
        buffer[i] = input_jpeg.buffer[i];
    }
#pragma acc enter data copyin(filteredImage[0 : width * height * num_channels], \
                              buffer[0 : width * height * num_channels], \
                              flattened_filter[FILTER_SIZE * FILTER_SIZE])

#pragma acc update device(filteredImage[0 : width * height * num_channels], \
                              buffer[0 : width * height * num_channels], \
                              flattened_filter[FILTER_SIZE * FILTER_SIZE])

    auto start_time = std::chrono::high_resolution_clock::now();
#pragma acc parallel present(filteredImage[0 : width * height * num_channels], \
                              buffer[0 : width * height * num_channels], \
                              flattened_filter[FILTER_SIZE * FILTER_SIZE]) \
    num_gangs(1024)
    {
#pragma acc loop collapse(2)
        for (int row = 1; row < height - 1 ; row++){
            for (int col = 1; col < width - 1; col++){


                unsigned char sum_r = 0,sum_g = 0,sum_b = 0;
                int channel_value_r, channel_value_g, channel_value_b;
                int index = (row * width + col) * num_channels;

                channel_value_r = buffer[((row - 1) * width + (col - 1)) * num_channels];
                channel_value_g = buffer[((row - 1) * width + (col - 1)) * num_channels + 1];
                channel_value_b = buffer[((row - 1) * width + (col - 1)) * num_channels + 2];
                sum_r += static_cast<unsigned char>(channel_value_r * flattened_filter[0]);
                sum_g += static_cast<unsigned char>(channel_value_g * flattened_filter[0]);
                sum_b += static_cast<unsigned char>(channel_value_b * flattened_filter[0]);


                channel_value_r = buffer[((row - 1) * width + col) * num_channels];
                channel_value_g = buffer[((row - 1) * width + col) * num_channels + 1];
                channel_value_b = buffer[((row - 1) * width + col) * num_channels + 2];
                sum_r += static_cast<unsigned char>(channel_value_r * flattened_filter[1]);
                sum_g += static_cast<unsigned char>(channel_value_g * flattened_filter[1]);
                sum_b += static_cast<unsigned char>(channel_value_b * flattened_filter[1]);

                channel_value_r = buffer[((row - 1) * width + (col + 1)) * num_channels];
                channel_value_g = buffer[((row - 1) * width + (col + 1)) * num_channels + 1];
                channel_value_b = buffer[((row - 1) * width + (col + 1)) * num_channels + 2];
                sum_r += static_cast<unsigned char>(channel_value_r * flattened_filter[2]);
                sum_g += static_cast<unsigned char>(channel_value_g * flattened_filter[2]);
                sum_b += static_cast<unsigned char>(channel_value_b * flattened_filter[2]);

                channel_value_r = buffer[(row * width + (col - 1)) * num_channels];
                channel_value_g = buffer[(row * width + (col - 1)) * num_channels + 1];
                channel_value_b = buffer[(row * width + (col - 1)) * num_channels + 2];
                sum_r += static_cast<unsigned char>(channel_value_r * flattened_filter[3]);
                sum_g += static_cast<unsigned char>(channel_value_g * flattened_filter[3]);
                sum_b += static_cast<unsigned char>(channel_value_b * flattened_filter[3]);

                channel_value_r = buffer[(row * width + col) * num_channels];
                channel_value_g = buffer[(row * width + col) * num_channels + 1];
                channel_value_b = buffer[(row * width + col) * num_channels + 2];
                sum_r += static_cast<unsigned char>(channel_value_r * flattened_filter[4]);
                sum_g += static_cast<unsigned char>(channel_value_g * flattened_filter[4]);
                sum_b += static_cast<unsigned char>(channel_value_b * flattened_filter[4]);

                channel_value_r = buffer[(row * width + (col + 1)) * num_channels];
                channel_value_g = buffer[(row * width + (col + 1)) * num_channels + 1];
                channel_value_b = buffer[(row * width + (col + 1)) * num_channels + 2];
                sum_r += static_cast<unsigned char>(channel_value_r * flattened_filter[5]);
                sum_g += static_cast<unsigned char>(channel_value_g * flattened_filter[5]);
                sum_b += static_cast<unsigned char>(channel_value_b * flattened_filter[5]);

                channel_value_r = buffer[((row + 1) * width + (col - 1)) * num_channels];
                channel_value_g = buffer[((row + 1) * width + (col - 1)) * num_channels + 1];
                channel_value_b = buffer[((row + 1) * width + (col - 1)) * num_channels + 2];
                sum_r += static_cast<unsigned char>(channel_value_r * flattened_filter[6]);
                sum_g += static_cast<unsigned char>(channel_value_g * flattened_filter[6]);
                sum_b += static_cast<unsigned char>(channel_value_b * flattened_filter[6]);

                channel_value_r = buffer[((row + 1) * width + col) * num_channels];
                channel_value_g = buffer[((row + 1) * width + col) * num_channels + 1];
                channel_value_b = buffer[((row + 1) * width + col) * num_channels + 2];
                sum_r += static_cast<unsigned char>(channel_value_r * flattened_filter[7]);
                sum_g += static_cast<unsigned char>(channel_value_g * flattened_filter[7]);
                sum_b += static_cast<unsigned char>(channel_value_b * flattened_filter[7]);

                channel_value_r = buffer[((row + 1) * width + (col + 1)) * num_channels];
                channel_value_g = buffer[((row + 1) * width + (col + 1)) * num_channels + 1];
                channel_value_b = buffer[((row + 1) * width + (col + 1)) * num_channels + 2];
                sum_r += static_cast<unsigned char>(channel_value_r * flattened_filter[8]);
                sum_g += static_cast<unsigned char>(channel_value_b * flattened_filter[8]);
                sum_b += static_cast<unsigned char>(channel_value_b * flattened_filter[8]);

                filteredImage[index] = sum_r;
                filteredImage[index + 1] = sum_g;
                filteredImage[index + 2] = sum_b;


            }
        }
    }              

    auto end_time = std::chrono::high_resolution_clock::now();
#pragma acc update self(filteredImage[0 : width * height * num_channels], \
                        buffer[0 : width * height * num_channels])

#pragma acc exit data copyout(filteredImage[0 : width * height * num_channels])

    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Write filteredImage to output JPEG
    const char *output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    delete[] buffer;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
