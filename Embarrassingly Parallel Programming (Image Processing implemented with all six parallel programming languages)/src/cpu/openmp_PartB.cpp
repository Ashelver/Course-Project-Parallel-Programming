// 120090712
// OpenMP implementation of applying a 3x3 low-pass filter to a JPEG image
//

#include <iostream>
#include <chrono>
#include <omp.h>    // OpenMP header
#include "utils.hpp"



const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // User-specified thread count
    int num_threads = std::stoi(argv[3]); 
    // Read input JPEG image
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
     
    int num_rows = input_jpeg.height;
    int num_cols = input_jpeg.width;

    // Separate R, G, B channels into three continuous arrays
    auto rChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto gChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    auto bChannel = new unsigned char[input_jpeg.width * input_jpeg.height];
    
    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        rChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        gChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        bChannel[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }

    // applying a 3x3 low-pass filter in parallel
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for default(none) shared(filter, num_rows, num_cols, rChannel, gChannel, bChannel, filteredImage, input_jpeg) num_threads(num_threads)
    for (int height = 0; height < num_rows; height++)
    {
        for (int width = 0; width < num_cols; width++)
        {

            int channel_value_r = rChannel[(height) * num_cols + (width)];
            int channel_value_g = gChannel[(height) * num_cols + (width)];
            int channel_value_b = bChannel[(height) * num_cols + (width)];         

            /*--------------------------------------------------------------------------------------------------------*/
            // UL
            if ((height - 1) >= 0 && (width - 1) >= 0){
                filteredImage[((height - 1) * num_cols + width - 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[2][2]);
                filteredImage[((height - 1) * num_cols + width - 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[2][2]);
                filteredImage[((height - 1) * num_cols + width - 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[2][2]);
            }


            // UM
            if ((height - 1) >= 0){
                filteredImage[((height - 1) * num_cols + width) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[2][1]);
                filteredImage[((height - 1) * num_cols + width) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[2][1]);
                filteredImage[((height - 1) * num_cols + width) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[2][1]);
            }


            // UR
            if ((height - 1) >= 0 && (width + 1) < num_cols){
                filteredImage[((height - 1) * num_cols + width + 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[2][0]);
                filteredImage[((height - 1) * num_cols + width + 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[2][0]);
                filteredImage[((height - 1) * num_cols + width + 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[2][0]);
            }

            /*--------------------------------------------------------------------------------------------------------*/
            // ML
            if ((width - 1) >= 0){
                filteredImage[(height * num_cols + width - 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[1][2]);
                filteredImage[(height * num_cols + width - 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[1][2]);
                filteredImage[(height * num_cols + width - 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[1][2]);
            }


            // MM
            if (true){
                filteredImage[(height * num_cols + width) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[1][1]);
                filteredImage[(height * num_cols + width) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[1][1]);
                filteredImage[(height * num_cols + width) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[1][1]);
            }


            // MR
            if ((width + 1) < num_cols){
                filteredImage[(height * num_cols + width + 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[1][0]);
                filteredImage[(height * num_cols + width + 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[1][0]);
                filteredImage[(height * num_cols + width + 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[1][0]);
            }


            /*--------------------------------------------------------------------------------------------------------*/
            // DL
            if ((height + 1) < num_rows && (width - 1) >= 0){
                filteredImage[((height + 1) * num_cols + width - 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[0][2]);
                filteredImage[((height + 1) * num_cols + width - 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[0][2]);
                filteredImage[((height + 1) * num_cols + width - 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[0][2]);
            }


            // DM
            if ((height + 1) < num_rows){
                filteredImage[((height + 1) * num_cols + width) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[0][1]);
                filteredImage[((height + 1) * num_cols + width) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[0][1]);
                filteredImage[((height + 1) * num_cols + width) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[0][1]);
            }


            // DR
            if ((height + 1) < num_rows && (width + 1) < num_cols){
                filteredImage[((height + 1) * num_cols + width + 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[0][0]);
                filteredImage[((height + 1) * num_cols + width + 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[0][0]);
                filteredImage[((height + 1) * num_cols + width + 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[0][0]);
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output JPEG filtered image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to save output JPEG image\n";
        return -1;
    }

    // Release the allocated memory
    delete[] input_jpeg.buffer;
    delete[] rChannel;
    delete[] gChannel;
    delete[] bChannel;
    delete[] filteredImage;
    
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}
