// 120090712
// Parallel Merge Sort
//

#include <iostream>
#include <vector>
#include <chrono>
#include <pthread.h> // pthread header
#include <omp.h> // openmp header
#include "../utils.hpp"



// double findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2) {
//     if (nums1.size() > nums2.size()) swap(nums1, nums2);
//     int M = nums1.size(), N = nums2.size(), L = 0, R = M, K = (M + N + 1) / 2;
//     while (true) {
//         int i = (L + R) / 2, j = K - i;
//         if (i < M && nums2[j - 1] > nums1[i]) L = i + 1;
//         else if (i > L && nums1[i - 1] > nums2[j]) R = i - 1;
//         else {
//             int maxLeft = max(i ? nums1[i - 1] : INT_MIN, j ? nums2[j - 1] : INT_MIN);
//             if ((M + N) % 2) return maxLeft;
//             int minRight = min(i == M ? INT_MAX : nums1[i], j == N ? INT_MAX : nums2[j]);
//             return (maxLeft + minRight) / 2.0;
//         }
//     }
// }




void merge(std::vector<int>& vec, int l, int m, int r) {
    /* Your code here!
       Implement parallel merge algorithm
    */
}

void mergeSort(std::vector<int>& vec, int l, int r) {
    /* Your code here!
       Implement parallel merge sort by dynamic threads creation
    */
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable threads_num vector_size\n"
            );
    }

    const int thread_num = atoi(argv[1]);

    const int size = atoi(argv[2]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    std::vector<int> S(size);
    std::vector<int> L(size);
    std::vector<int> results(size);

    auto start_time = std::chrono::high_resolution_clock::now();

    mergeSort(vec, 0, size - 1);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Merge Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);
}