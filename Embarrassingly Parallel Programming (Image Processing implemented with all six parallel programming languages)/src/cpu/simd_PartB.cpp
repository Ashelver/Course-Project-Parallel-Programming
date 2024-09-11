// 120090712
// SIMD (AVX2) implementation of applying a 3x3 low-pass filter to a JPEG image
//

#include <iostream>
#include <chrono>

#include <immintrin.h>

#include "utils.hpp"



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
    // Read JPEG File
    const char* input_filepath = argv[1];
    std::cout << "Input file from: " << input_filepath << "\n";
    auto input_jpeg = read_from_jpeg(input_filepath);
    if (input_jpeg.buffer == NULL) {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    // Apply the filter to the image
    auto filteredImage = new unsigned char[input_jpeg.width * input_jpeg.height * input_jpeg.num_channels + 8];
    for (int i = 0; i < input_jpeg.width * input_jpeg.height * input_jpeg.num_channels; ++i)
        filteredImage[i] = 0;

    // Prepross, store reds, greens and blues separately
    auto reds = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto greens = new unsigned char[input_jpeg.width * input_jpeg.height + 16];
    auto blues = new unsigned char[input_jpeg.width * input_jpeg.height + 16];

    for (int i = 0; i < input_jpeg.width * input_jpeg.height; i++) {
        reds[i] = input_jpeg.buffer[i * input_jpeg.num_channels];
        greens[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 1];
        blues[i] = input_jpeg.buffer[i * input_jpeg.num_channels + 2];
    }


    // Set SIMD scalars, we use AVX2 instructions
    __m256 ULScalar = _mm256_set1_ps(filter[0][0]);
    __m256 UMScalar = _mm256_set1_ps(filter[0][1]);
    __m256 URScalar = _mm256_set1_ps(filter[0][2]);
    __m256 MLScalar = _mm256_set1_ps(filter[1][0]);
    __m256 MMScalar = _mm256_set1_ps(filter[1][1]);
    __m256 MRScalar = _mm256_set1_ps(filter[1][2]);
    __m256 DLScalar = _mm256_set1_ps(filter[2][0]);
    __m256 DMScalar = _mm256_set1_ps(filter[2][1]);
    __m256 DRScalar = _mm256_set1_ps(filter[2][2]);


    // Mask used for moving the bits
    // |0|0|0|8|0|0|0|7|0|0|0|6|0|0|0|5|0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> 
    // |0|0|0|0|0|0|0|0|0|0|8|0|0|7|0|0|6|0|0|5|0|0|4|0|0|3|0|0|2|0|0|1|
    __m128i shuffle_red = _mm_setr_epi8(0, -1, -1, 4, 
                                    -1, -1, 8, -1, 
                                    -1, 12, -1, -1, 
                                    -1, -1, -1, -1);


    // |0|0|0|8|0|0|0|7|0|0|0|6|0|0|0|5|0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> 
    // |0|0|0|0|0|0|0|0|0|8|0|0|7|0|0|6|0|0|5|0|0|4|0|0|3|0|0|2|0|0|1|0|
    __m128i shuffle_green = _mm_setr_epi8(-1, 0, -1, -1, 
                                    4, -1, -1, 8, 
                                    -1, -1, 12, -1, 
                                    -1, -1, -1, -1);

    // |0|0|0|8|0|0|0|7|0|0|0|6|0|0|0|5|0|0|0|4|0|0|0|3|0|0|0|2|0|0|0|1| -> 
    // |0|0|0|0|0|0|0|0|8|0|0|7|0|0|6|0|0|5|0|0|4|0|0|3|0|0|2|0|0|1|0|0|
    __m128i shuffle_blue = _mm_setr_epi8(-1, -1, 0, -1, 
                                    -1, 4, -1, -1, 
                                    8, -1, -1, 12, 
                                    -1, -1, -1, -1);


    // Ignore the pixels of the vertical edges
    // Using SIMD to accelerate the transformation
    auto start_time = std::chrono::high_resolution_clock::now();    // Start recording time
    for (int i = input_jpeg.width; i < input_jpeg.width * (input_jpeg.height - 1); i+=8) {

        int ul = i - (input_jpeg.width + 1);
        int um = i - (input_jpeg.width );
        int ur = i - (input_jpeg.width - 1);

        int ml = i - 1;
        int mm = i;
        int mr = i + 1;

        int dl = i + (input_jpeg.width - 1);
        int dm = i + (input_jpeg.width );
        int dr = i + (input_jpeg.width + 1);

        // Initialize zeros results
        __m256 red_results = _mm256_setzero_ps();
        __m256 green_results = _mm256_setzero_ps();
        __m256 blue_results = _mm256_setzero_ps();


        // --------------------------- UL -------------------------------
        // Load the 8 red chars to a 256 bits float register
        __m128i red_chars = _mm_loadu_si128((__m128i*) (reds+ul));
        __m256i red_ints = _mm256_cvtepu8_epi32(red_chars);
        __m256 red_floats = _mm256_cvtepi32_ps(red_ints);
        // Multiply the red floats to the red scalar
        red_results = _mm256_add_ps(_mm256_mul_ps(red_floats, ULScalar),red_results);

        // Load the 8 green chars to a 256 bits float register
        __m128i green_chars = _mm_loadu_si128((__m128i*) (greens+ul));
        __m256i green_ints = _mm256_cvtepu8_epi32(green_chars);
        __m256 green_floats = _mm256_cvtepi32_ps(green_ints);
        // Multiply the green floats to the green scalar
        green_results = _mm256_add_ps(_mm256_mul_ps(green_floats, ULScalar),green_results);

        // Load the 8 blue chars to a 256 bits float register
        __m128i blue_chars = _mm_loadu_si128((__m128i*) (blues+ul));
        __m256i blue_ints = _mm256_cvtepu8_epi32(blue_chars);
        __m256 blue_floats = _mm256_cvtepi32_ps(blue_ints);
        // Multiply the blue floats to the blue scalar
        blue_results = _mm256_add_ps(_mm256_mul_ps(blue_floats, ULScalar),blue_results);


        // --------------------------- UM -------------------------------
        // Load the 8 red chars to a 256 bits float register
        red_chars = _mm_loadu_si128((__m128i*) (reds+um));
        red_ints = _mm256_cvtepu8_epi32(red_chars);
        red_floats = _mm256_cvtepi32_ps(red_ints);
        // Multiply the red floats to the red scalar
        red_results = _mm256_add_ps(_mm256_mul_ps(red_floats, UMScalar),red_results);

        // Load the 8 green chars to a 256 bits float register
        green_chars = _mm_loadu_si128((__m128i*) (greens+um));
        green_ints = _mm256_cvtepu8_epi32(green_chars);
        green_floats = _mm256_cvtepi32_ps(green_ints);
        // Multiply the green floats to the green scalar
        green_results = _mm256_add_ps(_mm256_mul_ps(green_floats, UMScalar),green_results);

        // Load the 8 blue chars to a 256 bits float register
        blue_chars = _mm_loadu_si128((__m128i*) (blues+um));
        blue_ints = _mm256_cvtepu8_epi32(blue_chars);
        blue_floats = _mm256_cvtepi32_ps(blue_ints);
        // Multiply the blue floats to the blue scalar
        blue_results = _mm256_add_ps(_mm256_mul_ps(blue_floats, UMScalar),blue_results);


        // --------------------------- UR -------------------------------
        // Load the 8 red chars to a 256 bits float register
        red_chars = _mm_loadu_si128((__m128i*) (reds+ur));
        red_ints = _mm256_cvtepu8_epi32(red_chars);
        red_floats = _mm256_cvtepi32_ps(red_ints);
        // Multiply the red floats to the red scalar
        red_results = _mm256_add_ps(_mm256_mul_ps(red_floats, URScalar),red_results);

        // Load the 8 green chars to a 256 bits float register
        green_chars = _mm_loadu_si128((__m128i*) (greens+ur));
        green_ints = _mm256_cvtepu8_epi32(green_chars);
        green_floats = _mm256_cvtepi32_ps(green_ints);
        // Multiply the green floats to the green scalar
        green_results = _mm256_add_ps(_mm256_mul_ps(green_floats, URScalar),green_results);

        // Load the 8 blue chars to a 256 bits float register
        blue_chars = _mm_loadu_si128((__m128i*) (blues+ur));
        blue_ints = _mm256_cvtepu8_epi32(blue_chars);
        blue_floats = _mm256_cvtepi32_ps(blue_ints);
        // Multiply the blue floats to the blue scalar
        blue_results = _mm256_add_ps(_mm256_mul_ps(blue_floats, URScalar),blue_results);



        // --------------------------- ML -------------------------------
        // Load the 8 red chars to a 256 bits float register
        red_chars = _mm_loadu_si128((__m128i*) (reds+ml));
        red_ints = _mm256_cvtepu8_epi32(red_chars);
        red_floats = _mm256_cvtepi32_ps(red_ints);
        // Multiply the red floats to the red scalar
        red_results = _mm256_add_ps(_mm256_mul_ps(red_floats, MLScalar),red_results);

        // Load the 8 green chars to a 256 bits float register
        green_chars = _mm_loadu_si128((__m128i*) (greens+ml));
        green_ints = _mm256_cvtepu8_epi32(green_chars);
        green_floats = _mm256_cvtepi32_ps(green_ints);
        // Multiply the green floats to the green scalar
        green_results = _mm256_add_ps(_mm256_mul_ps(green_floats, MLScalar),green_results);

        // Load the 8 blue chars to a 256 bits float register
        blue_chars = _mm_loadu_si128((__m128i*) (blues+ml));
        blue_ints = _mm256_cvtepu8_epi32(blue_chars);
        blue_floats = _mm256_cvtepi32_ps(blue_ints);
        // Multiply the blue floats to the blue scalar
        blue_results = _mm256_add_ps(_mm256_mul_ps(blue_floats, MLScalar),blue_results);


        // --------------------------- MM -------------------------------
        // Load the 8 red chars to a 256 bits float register
        red_chars = _mm_loadu_si128((__m128i*) (reds+mm));
        red_ints = _mm256_cvtepu8_epi32(red_chars);
        red_floats = _mm256_cvtepi32_ps(red_ints);
        // Multiply the red floats to the red scalar
        red_results = _mm256_add_ps(_mm256_mul_ps(red_floats, MMScalar),red_results);

        // Load the 8 green chars to a 256 bits float register
        green_chars = _mm_loadu_si128((__m128i*) (greens+mm));
        green_ints = _mm256_cvtepu8_epi32(green_chars);
        green_floats = _mm256_cvtepi32_ps(green_ints);
        // Multiply the green floats to the green scalar
        green_results = _mm256_add_ps(_mm256_mul_ps(green_floats, MMScalar),green_results);

        // Load the 8 blue chars to a 256 bits float register
        blue_chars = _mm_loadu_si128((__m128i*) (blues+mm));
        blue_ints = _mm256_cvtepu8_epi32(blue_chars);
        blue_floats = _mm256_cvtepi32_ps(blue_ints);
        // Multiply the blue floats to the blue scalar
        blue_results = _mm256_add_ps(_mm256_mul_ps(blue_floats, MMScalar),blue_results);


        // --------------------------- MR -------------------------------
        // Load the 8 red chars to a 256 bits float register
        red_chars = _mm_loadu_si128((__m128i*) (reds+mr));
        red_ints = _mm256_cvtepu8_epi32(red_chars);
        red_floats = _mm256_cvtepi32_ps(red_ints);
        // Multiply the red floats to the red scalar
        red_results = _mm256_add_ps(_mm256_mul_ps(red_floats, MRScalar),red_results);

        // Load the 8 green chars to a 256 bits float register
        green_chars = _mm_loadu_si128((__m128i*) (greens+mr));
        green_ints = _mm256_cvtepu8_epi32(green_chars);
        green_floats = _mm256_cvtepi32_ps(green_ints);
        // Multiply the green floats to the green scalar
        green_results = _mm256_add_ps(_mm256_mul_ps(green_floats, MRScalar),green_results);

        // Load the 8 blue chars to a 256 bits float register
        blue_chars = _mm_loadu_si128((__m128i*) (blues+mr));
        blue_ints = _mm256_cvtepu8_epi32(blue_chars);
        blue_floats = _mm256_cvtepi32_ps(blue_ints);
        // Multiply the blue floats to the blue scalar
        blue_results = _mm256_add_ps(_mm256_mul_ps(blue_floats, MRScalar),blue_results);


        // --------------------------- DL -------------------------------
        // Load the 8 red chars to a 256 bits float register
        red_chars = _mm_loadu_si128((__m128i*) (reds+dl));
        red_ints = _mm256_cvtepu8_epi32(red_chars);
        red_floats = _mm256_cvtepi32_ps(red_ints);
        // Multiply the red floats to the red scalar
        red_results = _mm256_add_ps(_mm256_mul_ps(red_floats, DLScalar),red_results);

        // Load the 8 green chars to a 256 bits float register
        green_chars = _mm_loadu_si128((__m128i*) (greens+dl));
        green_ints = _mm256_cvtepu8_epi32(green_chars);
        green_floats = _mm256_cvtepi32_ps(green_ints);
        // Multiply the green floats to the green scalar
        green_results = _mm256_add_ps(_mm256_mul_ps(green_floats, DLScalar),green_results);

        // Load the 8 blue chars to a 256 bits float register
        blue_chars = _mm_loadu_si128((__m128i*) (blues+dl));
        blue_ints = _mm256_cvtepu8_epi32(blue_chars);
        blue_floats = _mm256_cvtepi32_ps(blue_ints);
        // Multiply the blue floats to the blue scalar
        blue_results = _mm256_add_ps(_mm256_mul_ps(blue_floats, DLScalar),blue_results);


        // --------------------------- DM -------------------------------
        // Load the 8 red chars to a 256 bits float register
        red_chars = _mm_loadu_si128((__m128i*) (reds+dm));
        red_ints = _mm256_cvtepu8_epi32(red_chars);
        red_floats = _mm256_cvtepi32_ps(red_ints);
        // Multiply the red floats to the red scalar
        red_results = _mm256_add_ps(_mm256_mul_ps(red_floats, DMScalar),red_results);

        // Load the 8 green chars to a 256 bits float register
        green_chars = _mm_loadu_si128((__m128i*) (greens+dm));
        green_ints = _mm256_cvtepu8_epi32(green_chars);
        green_floats = _mm256_cvtepi32_ps(green_ints);
        // Multiply the green floats to the green scalar
        green_results = _mm256_add_ps(_mm256_mul_ps(green_floats, DMScalar),green_results);

        // Load the 8 blue chars to a 256 bits float register
        blue_chars = _mm_loadu_si128((__m128i*) (blues+dm));
        blue_ints = _mm256_cvtepu8_epi32(blue_chars);
        blue_floats = _mm256_cvtepi32_ps(blue_ints);
        // Multiply the blue floats to the blue scalar
        blue_results = _mm256_add_ps(_mm256_mul_ps(blue_floats, DMScalar),blue_results);


        // --------------------------- DR -------------------------------
        // Load the 8 red chars to a 256 bits float register
        red_chars = _mm_loadu_si128((__m128i*) (reds+dr));
        red_ints = _mm256_cvtepu8_epi32(red_chars);
        red_floats = _mm256_cvtepi32_ps(red_ints);
        // Multiply the red floats to the red scalar
        red_results = _mm256_add_ps(_mm256_mul_ps(red_floats, DRScalar),red_results);

        // Load the 8 green chars to a 256 bits float register
        green_chars = _mm_loadu_si128((__m128i*) (greens+dr));
        green_ints = _mm256_cvtepu8_epi32(green_chars);
        green_floats = _mm256_cvtepi32_ps(green_ints);
        // Multiply the green floats to the green scalar
        green_results = _mm256_add_ps(_mm256_mul_ps(green_floats, DRScalar),green_results);

        // Load the 8 blue chars to a 256 bits float register
        blue_chars = _mm_loadu_si128((__m128i*) (blues+dr));
        blue_ints = _mm256_cvtepu8_epi32(blue_chars);
        blue_floats = _mm256_cvtepi32_ps(blue_ints);
        // Multiply the blue floats to the blue scalar
        blue_results = _mm256_add_ps(_mm256_mul_ps(blue_floats, DRScalar),blue_results);





        /*After the above steps: 
            color_results is [|32bit color1|32bit color2|32bit color3|32bit color4|32bit color5|32bit color6|32bit color7|32bit color8|]
        */ 

        // Transfer 32bit to 8bit
        __m256i red_results_chars = _mm256_cvtps_epi32(red_results);
        __m256i green_results_chars = _mm256_cvtps_epi32(green_results);
        __m256i blue_results_chars = _mm256_cvtps_epi32(blue_results);

        /*Combine the three colors
            1. move the bits for green and blue.
            2. add them
        */ 

        // Seperate the 256bits results to 2 128bits result
        __m128i red_low = _mm256_castsi256_si128(red_results_chars);
        __m128i red_high = _mm256_extracti128_si256(red_results_chars, 1);

        __m128i red_trans_low = _mm_shuffle_epi8(red_low, shuffle_red);
        __m128i red_trans_high = _mm_shuffle_epi8(red_high, shuffle_red);
 

        __m128i green_low = _mm256_castsi256_si128(green_results_chars);
        __m128i green_high = _mm256_extracti128_si256(green_results_chars, 1);

        __m128i green_trans_low = _mm_shuffle_epi8(green_low, shuffle_green);
        __m128i green_trans_high = _mm_shuffle_epi8(green_high, shuffle_green);


        __m128i blue_low = _mm256_castsi256_si128(blue_results_chars);
        __m128i blue_high = _mm256_extracti128_si256(blue_results_chars, 1);

        __m128i blue_trans_low = _mm_shuffle_epi8(blue_low, shuffle_blue);
        __m128i blue_trans_high = _mm_shuffle_epi8(blue_high, shuffle_blue);


        __m128i result_low = _mm_add_epi8(_mm_add_epi8(red_trans_low,green_trans_low),blue_trans_low);
        __m128i result_high = _mm_add_epi8(_mm_add_epi8(red_trans_high,green_trans_high),blue_trans_high);
        // // Store the data

        _mm_storeu_si128((__m128i*)(&filteredImage[i*input_jpeg.num_channels]), result_low);
        _mm_storeu_si128((__m128i*)(&filteredImage[(i+4)*input_jpeg.num_channels]), result_high);

    }


    auto end_time = std::chrono::high_resolution_clock::now();  // Stop recording time
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output filtered JPEG Image
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

