// 120090712
// A optimizd sequential implementation of image filtering
//

#include <iostream>
#include <cmath>
#include <chrono>

#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    auto input_jpeg = read_from_jpeg(input_filename);
    // Apply the filter to the image
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    // Optimized loop

    for (int height = 0; height < input_jpeg.height; height++)
    {
        for (int width = 0; width < input_jpeg.width; width++)
        {

            int channel_value_r = input_jpeg.buffer[((height) * input_jpeg.width + (width)) * input_jpeg.num_channels];
            int channel_value_g = input_jpeg.buffer[((height) * input_jpeg.width + (width)) * input_jpeg.num_channels + 1];
            int channel_value_b = input_jpeg.buffer[((height) * input_jpeg.width + (width)) * input_jpeg.num_channels + 2];         

            /*--------------------------------------------------------------------------------------------------------*/
            // UL
            if ((height - 1) >= 0 && (width - 1) >= 0){
                filteredImage[((height - 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[2][2]);
                filteredImage[((height - 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[2][2]);
                filteredImage[((height - 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[2][2]);
            }


            // UM
            if ((height - 1) >= 0){
                filteredImage[((height - 1) * input_jpeg.width + width) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[2][1]);
                filteredImage[((height - 1) * input_jpeg.width + width) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[2][1]);
                filteredImage[((height - 1) * input_jpeg.width + width) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[2][1]);
            }


            // UR
            if ((height - 1) >= 0 && (width + 1) < input_jpeg.width){
                filteredImage[((height - 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[2][0]);
                filteredImage[((height - 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[2][0]);
                filteredImage[((height - 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[2][0]);
            }

            /*--------------------------------------------------------------------------------------------------------*/
            // ML
            if ((width - 1) >= 0){
                filteredImage[(height * input_jpeg.width + width - 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[1][2]);
                filteredImage[(height * input_jpeg.width + width - 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[1][2]);
                filteredImage[(height * input_jpeg.width + width - 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[1][2]);
            }


            // MM
            if (true){
                filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[1][1]);
                filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[1][1]);
                filteredImage[(height * input_jpeg.width + width) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[1][1]);
            }


            // MR
            if ((width + 1) < input_jpeg.width){
                filteredImage[(height * input_jpeg.width + width + 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[1][0]);
                filteredImage[(height * input_jpeg.width + width + 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[1][0]);
                filteredImage[(height * input_jpeg.width + width + 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[1][0]);
            }


            /*--------------------------------------------------------------------------------------------------------*/
            // DL
            if ((height + 1) < input_jpeg.height && (width - 1) >= 0){
                filteredImage[((height + 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[0][2]);
                filteredImage[((height + 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[0][2]);
                filteredImage[((height + 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[0][2]);
            }


            // DM
            if ((height + 1) < input_jpeg.height){
                filteredImage[((height + 1) * input_jpeg.width + width) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[0][1]);
                filteredImage[((height + 1) * input_jpeg.width + width) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[0][1]);
                filteredImage[((height + 1) * input_jpeg.width + width) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[0][1]);
            }


            // DR
            if ((height + 1) < input_jpeg.height && (width + 1) < input_jpeg.width){
                filteredImage[((height + 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[0][0]);
                filteredImage[((height + 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[0][0]);
                filteredImage[((height + 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[0][0]);
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Post-processing
    delete[] input_jpeg.buffer;
    delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    return 0;
}