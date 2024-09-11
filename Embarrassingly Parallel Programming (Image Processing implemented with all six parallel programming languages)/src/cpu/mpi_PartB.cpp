// 120090712
// // MPI implementation of applying a 3x3 low-pass filter to a JPEG image
//

#include <iostream>
#include <vector>
#include <chrono>

#include <mpi.h>    // MPI Header

#include "utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

const int FILTER_SIZE = 3;
const double filter[FILTER_SIZE][FILTER_SIZE] = {
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9},
    {1.0 / 9, 1.0 / 9, 1.0 / 9}
};

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        std::cerr << "Invalid argument, should be: ./executable /path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
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

    // Read JPEG File
    const char * input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Divide the task
    // For a 4*3 graph and 3 tasks,
    // try to divide into 2*3, 1*3, 1*3
    int num_cols = input_jpeg.width;
    int num_rows = input_jpeg.height;
    int rows_per_task = num_rows / numtasks;
    int left_rows_num = num_rows % numtasks;

    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_rows_num = 0;

    for (int i = 0; i < numtasks; i++) {
        if (divided_left_rows_num < left_rows_num) {
            cuts[i+1] = cuts[i] + rows_per_task + 1;
            divided_left_rows_num++;
        } else cuts[i+1] = cuts[i] + rows_per_task;
    }


    // The tasks for the master executor
    // 1. Apply the low-pass filter to the RGB contents
    // 2. Receive the filtered contents from slave executors
    // 3. Write the filtered contents to the JPEG File
    if (taskid == MASTER) {
        int start;
        int end;
        if (cuts[MASTER] - 1 >= 0){
            start = cuts[MASTER] - 1;
        } else {
            start = cuts[MASTER];
        }
        if (cuts[MASTER + 1] + 1 <= num_rows){
            end = cuts[MASTER + 1] + 1;
        } else {
            end = cuts[MASTER + 1];
        }
        // Apply the low-pass filter to the RGB Contents
        auto filteredImage = new unsigned char[num_cols * num_rows * input_jpeg.num_channels];
        for (int i = 0; i < num_cols * num_rows * input_jpeg.num_channels; ++i)
            filteredImage[i] = 0;      
        for (int height = start; height < end; height++){
            for (int width = 0; width < num_cols; width++)
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
        


        // Receive the filtered contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; i++) {
            unsigned char* start_pos = filteredImage + cuts[i] * num_cols * input_jpeg.num_channels;
            int length = (cuts[i+1] - cuts[i]) * num_cols * input_jpeg.num_channels;
            MPI_Recv(start_pos, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        

        // Save the filtered Image
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JPEGMeta output_jpeg{filteredImage, input_jpeg.width, input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
        if (write_to_jpeg(output_jpeg, output_filepath)) {
            std::cerr << "Failed to write output JPEG to file\n";
            MPI_Finalize();
            return -1;
        }

        // Release the memory
        delete[] input_jpeg.buffer;
        delete[] filteredImage;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds\n";
    }
    // The tasks for the slave executor
    // 1. Apply the low-pass filter to the RGB contents
    // 2.
    // 3. Send the filtered contents back to the master executor
    else{
        // Apply the low-pass filter to the RGB Contents
        int start;
        int end;
        if (cuts[taskid] - 1 >= 0){
            start = cuts[taskid] - 1;
        } else {
            start = cuts[taskid];
        }
        if (cuts[taskid + 1] + 1 <= num_rows){
            end = cuts[taskid + 1] + 1;
        } else {
            end = cuts[taskid + 1];
        }
        // std::cout << start << end << std::endl;
        int filteredImage_rows = (cuts[taskid + 1] - cuts[taskid]);
        int length = filteredImage_rows * num_cols * input_jpeg.num_channels;

        auto filteredImage = new unsigned char[length];
        for (int i = 0; i < length; ++i)
            filteredImage[i] = 0;
        for (int height = start; height < end; height++){
            for (int width = 0; width < num_cols; width++)
            {
                int local_height = height - cuts[taskid];

                int channel_value_r = input_jpeg.buffer[((height) * input_jpeg.width + (width)) * input_jpeg.num_channels];
                int channel_value_g = input_jpeg.buffer[((height) * input_jpeg.width + (width)) * input_jpeg.num_channels + 1];
                int channel_value_b = input_jpeg.buffer[((height) * input_jpeg.width + (width)) * input_jpeg.num_channels + 2];     

                /*--------------------------------------------------------------------------------------------------------*/
                // UL
                if ((local_height - 1) >= 0  &&  (width - 1) >= 0){
                    filteredImage[((local_height - 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels]
                        += static_cast<unsigned char>(channel_value_r * filter[2][2]);
                    filteredImage[((local_height - 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels + 1]
                        += static_cast<unsigned char>(channel_value_g * filter[2][2]);
                    filteredImage[((local_height - 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels + 2]
                        += static_cast<unsigned char>(channel_value_b * filter[2][2]);
                }


                // UM
                if ((local_height - 1) >= 0){
                    filteredImage[((local_height - 1) * input_jpeg.width + width) * input_jpeg.num_channels]
                        += static_cast<unsigned char>(channel_value_r * filter[2][1]);
                    filteredImage[((local_height - 1) * input_jpeg.width + width) * input_jpeg.num_channels + 1]
                        += static_cast<unsigned char>(channel_value_g * filter[2][1]);
                    filteredImage[((local_height - 1) * input_jpeg.width + width) * input_jpeg.num_channels + 2]
                        += static_cast<unsigned char>(channel_value_b * filter[2][1]);
                }


                // UR
                if ((local_height - 1) >= 0 && (width + 1) < input_jpeg.width){
                    filteredImage[((local_height - 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels]
                        += static_cast<unsigned char>(channel_value_r * filter[2][0]);
                    filteredImage[((local_height - 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels + 1]
                        += static_cast<unsigned char>(channel_value_g * filter[2][0]);
                    filteredImage[((local_height - 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels + 2]
                        += static_cast<unsigned char>(channel_value_b * filter[2][0]);
                }

                /*--------------------------------------------------------------------------------------------------------*/
                // ML
                if (local_height  >= 0 && local_height  < filteredImage_rows && (width - 1) >= 0){
                    filteredImage[(local_height * input_jpeg.width + width - 1) * input_jpeg.num_channels]
                        += static_cast<unsigned char>(channel_value_r * filter[1][2]);
                    filteredImage[(local_height * input_jpeg.width + width - 1) * input_jpeg.num_channels + 1]
                        += static_cast<unsigned char>(channel_value_g * filter[1][2]);
                    filteredImage[(local_height * input_jpeg.width + width - 1) * input_jpeg.num_channels + 2]
                        += static_cast<unsigned char>(channel_value_b * filter[1][2]);
                }


                // MM
                if (local_height  >= 0 && local_height  < filteredImage_rows){
                    filteredImage[(local_height * input_jpeg.width + width) * input_jpeg.num_channels]
                        += static_cast<unsigned char>(channel_value_r * filter[1][1]);
                    filteredImage[(local_height * input_jpeg.width + width) * input_jpeg.num_channels + 1]
                        += static_cast<unsigned char>(channel_value_g * filter[1][1]);
                    filteredImage[(local_height * input_jpeg.width + width) * input_jpeg.num_channels + 2]
                        += static_cast<unsigned char>(channel_value_b * filter[1][1]);
                }


                // MR
                if (local_height  >= 0 && local_height  < filteredImage_rows && (width + 1) < input_jpeg.width){
                    filteredImage[(local_height * input_jpeg.width + width + 1) * input_jpeg.num_channels]
                        += static_cast<unsigned char>(channel_value_r * filter[1][0]);
                    filteredImage[(local_height * input_jpeg.width + width + 1) * input_jpeg.num_channels + 1]
                        += static_cast<unsigned char>(channel_value_g * filter[1][0]);
                    filteredImage[(local_height * input_jpeg.width + width + 1) * input_jpeg.num_channels + 2]
                        += static_cast<unsigned char>(channel_value_b * filter[1][0]);
                }


                /*--------------------------------------------------------------------------------------------------------*/
                // DL
                if ((local_height + 1) < filteredImage_rows && (width - 1) >= 0){
                    filteredImage[((local_height + 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels]
                        += static_cast<unsigned char>(channel_value_r * filter[0][2]);
                    filteredImage[((local_height + 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels + 1]
                        += static_cast<unsigned char>(channel_value_g * filter[0][2]);
                    filteredImage[((local_height + 1) * input_jpeg.width + width - 1) * input_jpeg.num_channels + 2]
                        += static_cast<unsigned char>(channel_value_b * filter[0][2]);
                }


                // DM
                if ((local_height + 1) < filteredImage_rows){
                    filteredImage[((local_height + 1) * input_jpeg.width + width) * input_jpeg.num_channels]
                        += static_cast<unsigned char>(channel_value_r * filter[0][1]);
                    filteredImage[((local_height + 1) * input_jpeg.width + width) * input_jpeg.num_channels + 1]
                        += static_cast<unsigned char>(channel_value_g * filter[0][1]);
                    filteredImage[((local_height + 1) * input_jpeg.width + width) * input_jpeg.num_channels + 2]
                        += static_cast<unsigned char>(channel_value_b * filter[0][1]);
                }


                // DR
                if ((local_height + 1) < filteredImage_rows && (width + 1) < input_jpeg.width){
                    filteredImage[((local_height + 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels]
                        += static_cast<unsigned char>(channel_value_r * filter[0][0]);
                    filteredImage[((local_height + 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels + 1]
                        += static_cast<unsigned char>(channel_value_g * filter[0][0]);
                    filteredImage[((local_height + 1) * input_jpeg.width + width + 1) * input_jpeg.num_channels + 2]
                        += static_cast<unsigned char>(channel_value_b * filter[0][0]);
                }
            }            
        }
        // Send the filtered Image back to the master
        MPI_Send(filteredImage, length, MPI_CHAR, MASTER, TAG_GATHER, MPI_COMM_WORLD);

        
        // Release the memory
        delete[] filteredImage;
    }

}