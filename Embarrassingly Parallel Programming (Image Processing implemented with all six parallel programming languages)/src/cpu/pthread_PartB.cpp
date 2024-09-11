// 120090712
// // Pthread implementation of applying a 3x3 low-pass filter to a JPEG image
//



#include <iostream>
#include <cmath>
#include <chrono>
#include <pthread.h>
#include "utils.hpp"

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

// Declare a barrier
pthread_barrier_t barrier;

// Structure to hold thread arguments
struct ThreadArgs {
    unsigned char* input_buffer;
    unsigned char* output_buffer;
    int width;
    int height;
    int num_channels;
    int start_row;
    int end_row;
};

// Function to be executed by each thread
void* applyFilter(void* thread_args) {
    ThreadArgs* args = reinterpret_cast<ThreadArgs*>(thread_args);
    int start_row;
    int end_row;

    if (args->start_row - 1 >= 0){
        start_row = args->start_row - 1;
    } else {
        start_row = args->start_row;
    }

    if (args->end_row + 1 <= args->height){
        end_row = args->end_row + 1;
    } else {
        end_row = args->end_row;
    }

    int filteredImage_rows = args->end_row - args->start_row;
    int length = filteredImage_rows * args->width * args->num_channels;
    auto short_filteredImage = new unsigned char[length];
    for (int height = start_row; height <end_row; height++){
        for (int width = 0; width < args->width; width++)
        {
            int local_height = height - args->start_row;
            int channel_value_r = args->input_buffer[((height) * args->width + (width)) * args->num_channels];
            int channel_value_g = args->input_buffer[((height) * args->width + (width)) * args->num_channels + 1];
            int channel_value_b = args->input_buffer[((height) * args->width + (width)) * args->num_channels + 2]; 
            /*--------------------------------------------------------------------------------------------------------*/

            // UL
            if ((local_height - 1) >= 0  &&  (width - 1) >= 0){
                short_filteredImage[((local_height - 1) * args->width + width - 1) * args->num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[2][2]);
                short_filteredImage[((local_height - 1) * args->width + width - 1) * args->num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[2][2]);
                short_filteredImage[((local_height - 1) * args->width + width - 1) * args->num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[2][2]);
            }


            // UM
            if ((local_height - 1) >= 0){
                short_filteredImage[((local_height - 1) * args->width + width) * args->num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[2][1]);
                short_filteredImage[((local_height - 1) * args->width + width) * args->num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[2][1]);
                short_filteredImage[((local_height - 1) * args->width + width) * args->num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[2][1]);
            }


            // UR
            if ((local_height - 1) >= 0 && (width + 1) < args->width){
                short_filteredImage[((local_height - 1) * args->width + width + 1) * args->num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[2][0]);
                short_filteredImage[((local_height - 1) * args->width + width + 1) * args->num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[2][0]);
                short_filteredImage[((local_height - 1) * args->width + width + 1) * args->num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[2][0]);
            }

            /*--------------------------------------------------------------------------------------------------------*/
            // ML
            if (local_height  >= 0 && local_height  < filteredImage_rows &&(width - 1) >= 0){
                short_filteredImage[(local_height * args->width + width - 1) * args->num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[1][2]);
                short_filteredImage[(local_height * args->width + width - 1) * args->num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[1][2]);
                short_filteredImage[(local_height * args->width + width - 1) * args->num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[1][2]);
            }


            // MM
            if (local_height  >= 0 && local_height  < filteredImage_rows){
                short_filteredImage[(local_height * args->width + width) * args->num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[1][1]);
                short_filteredImage[(local_height * args->width + width) * args->num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[1][1]);
                short_filteredImage[(local_height * args->width + width) * args->num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[1][1]);
            }


            // MR
            if (local_height  >= 0 && local_height  < filteredImage_rows && (width + 1) < args->width){
                short_filteredImage[(local_height * args->width + width + 1) * args->num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[1][0]);
                short_filteredImage[(local_height * args->width + width + 1) * args->num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[1][0]);
                short_filteredImage[(local_height * args->width + width + 1) * args->num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[1][0]);
            }


            /*--------------------------------------------------------------------------------------------------------*/
            // DL
            if ((local_height + 1) < filteredImage_rows && (width - 1) >= 0){
                short_filteredImage[((local_height + 1) * args->width + width - 1) * args->num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[0][2]);
                short_filteredImage[((local_height + 1) * args->width + width - 1) * args->num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[0][2]);
                short_filteredImage[((local_height + 1) * args->width + width - 1) * args->num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[0][2]);
            }


            // DM
            if ((local_height + 1) < filteredImage_rows){
                short_filteredImage[((local_height + 1) * args->width + width) * args->num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[0][1]);
                short_filteredImage[((local_height + 1) * args->width + width) * args->num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[0][1]);
                short_filteredImage[((local_height + 1) * args->width + width) * args->num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[0][1]);
            }

            // DR
            if ((local_height + 1) < filteredImage_rows && (width + 1) < args->width){
                short_filteredImage[((local_height + 1) * args->width + width + 1) * args->num_channels]
                    += static_cast<unsigned char>(channel_value_r * filter[0][0]);
                short_filteredImage[((local_height + 1) * args->width + width + 1) * args->num_channels + 1]
                    += static_cast<unsigned char>(channel_value_g * filter[0][0]);
                short_filteredImage[((local_height + 1) * args->width + width + 1) * args->num_channels + 2]
                    += static_cast<unsigned char>(channel_value_b * filter[0][0]);
            }
        }
    }
    // write back the result
    for (int height = args->start_row; height < args->end_row; height++){
        for (int width = 0; width < args->width; width++){
            args->output_buffer[(height * args->width + width) * args->num_channels] = static_cast<unsigned char>(short_filteredImage[((height - args->start_row) * args->width + width) * args->num_channels]);
            args->output_buffer[(height * args->width + width) * args->num_channels + 1] = static_cast<unsigned char>(short_filteredImage[((height - args->start_row) * args->width + width) * args->num_channels + 1]);
            args->output_buffer[(height * args->width + width) * args->num_channels + 2] = static_cast<unsigned char>(short_filteredImage[((height - args->start_row) * args->width + width) * args->num_channels + 2]);            
        }
    }
    pthread_barrier_wait(&barrier);
    delete[] short_filteredImage;
    return nullptr;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg num_threads\n";
        return -1;
    }


    // Create an array of pthreads
    int num_threads = std::stoi(argv[3]); // User-specified thread count

    // Initialize the barrier
    pthread_barrier_init(&barrier, NULL, num_threads);

    // Read from input JPEG
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);

    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels];

    pthread_t threads[num_threads];
    ThreadArgs thread_args[num_threads];

    auto start_time = std::chrono::high_resolution_clock::now();


    int rows_per_task = input_jpeg.height / num_threads;
    int left_rows_num = input_jpeg.height % num_threads;
    int index[num_threads + 1];
    for (int i = 0; i < num_threads + 1; i++){
        index[i] = 0;
    }
    int divided_left_rows_num = 0;

    
    for (int i = 0; i < num_threads; i++) {
        if (divided_left_rows_num < left_rows_num) {
            index[i+1] = index[i] + rows_per_task + 1;
            divided_left_rows_num++;
        } else index[i+1] = index[i] + rows_per_task;
    }

    
    // Divide the work among threads
    for (int i = 0; i < num_threads; i++) {
        thread_args[i].input_buffer = input_jpeg.buffer;
        thread_args[i].output_buffer = filteredImage;
        thread_args[i].width = input_jpeg.width;
        thread_args[i].height = input_jpeg.height;
        thread_args[i].num_channels = input_jpeg.num_channels;
        thread_args[i].start_row = index[i];
        thread_args[i].end_row = index[i + 1];

        int result = pthread_create(&threads[i], nullptr, applyFilter, &thread_args[i]);
        if (result != 0) {
            std::cerr << "Failed to create thread: " << result << std::endl;
            return -1;
        }
    }

    // Wait for all threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Write filteredImage to output JPEG
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (write_to_jpeg(output_jpeg, output_filepath)) {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Release allocated memory
    delete[] input_jpeg.buffer;
    delete[] filteredImage;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";

    return 0;

}
