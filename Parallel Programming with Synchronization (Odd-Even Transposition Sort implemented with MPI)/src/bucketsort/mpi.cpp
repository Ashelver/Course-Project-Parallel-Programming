// 120090712
//
// Parallel Bucket Sort with MPI
//

#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0


void show(std::vector<int> &vec, int start, int end){
    for (int i = start; i < end; ++i) {
        std::cout << vec[i] << ' ';
    }
    std::cout << std::endl;
}


void insertionSort(std::vector<int>& bucket) {

    for (int i = 1; i < bucket.size(); ++i) {
        int key = bucket[i];
        int j = i - 1;

        while (j >= 0 && bucket[j] > key) {
            bucket[j + 1] = bucket[j];
            j--;
        }

        bucket[j + 1] = key;
    }
}



void bucketSort(std::vector<int>& vec, int num_buckets, int numtasks, int taskid, MPI_Status* status) {
    // Optimal number of buckets: 
    int max_val = *std::max_element(vec.begin(), vec.end());
    int min_val = *std::min_element(vec.begin(), vec.end());
    // Divide the tasks
    int buckets_per_proc = num_buckets / numtasks;
    int left_buckets = num_buckets % numtasks;
    
    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_buckets = 0;

    for (int i = 0; i < numtasks; i++) {
        if (divided_left_buckets < left_buckets) {
            cuts[i+1] = cuts[i] + buckets_per_proc + 1;
            divided_left_buckets++;
        } else cuts[i+1] = cuts[i] + buckets_per_proc;
    }

    

    int range = max_val - min_val + 1;
    int small_bucket_size = range / num_buckets;
    int large_bucket_size = small_bucket_size + 1;
    int large_bucket_num = range - small_bucket_size * num_buckets;
    int boundary = min_val + large_bucket_num * large_bucket_size;

    std::vector<std::vector<int>> buckets(cuts[taskid + 1] - cuts[taskid]);
    // Pre-allocate space to avoid re-allocation
    for (std::vector<int>& bucket : buckets) {
        bucket.reserve(large_bucket_size);
    }

    // Place each element in the appropriate bucket
    int small_in_buckets;
    int large_in_buckets;
    if (cuts[taskid] < large_bucket_num){
        small_in_buckets = cuts[taskid]*large_bucket_size + min_val;
    } else {
        small_in_buckets = (cuts[taskid] - large_bucket_num)*small_bucket_size + boundary;
    }

    if (cuts[taskid + 1] < large_bucket_num){
        large_in_buckets = cuts[taskid + 1]*large_bucket_size + min_val;
    } else {
        large_in_buckets = (cuts[taskid + 1] - large_bucket_num)*small_bucket_size + boundary;
    }



    for (int num : vec) {
        if (num < small_in_buckets || num >= large_in_buckets){
            // Save the time
            continue;
        }
        int index;
        if (num < boundary) {
            index = (num - min_val) / large_bucket_size;
        } else {
            index = large_bucket_num + (num - boundary) / small_bucket_size;
        }
        if (index >= num_buckets) {
            // Handle elements at the upper bound
            index = num_buckets - 1;
        }
        if ( index >= cuts[taskid] && cuts[taskid + 1] > index ) buckets[index - cuts[taskid]].push_back(num);
    }

    for (std::vector<int>& bucket : buckets) {
        insertionSort(bucket);
    }


    std::vector<int> temp;
    for (std::vector<int>& bucket : buckets){
        temp.insert(temp.end(), bucket.begin(), bucket.end());
    }


    if (taskid == MASTER){
        int index = temp.size();
        temp.resize(vec.size());
        for (int i = MASTER + 1; i < numtasks; i++) {
            int t = 0;
            MPI_Recv(&t, 1, MPI_INT, i, 0, MPI_COMM_WORLD, status);
            MPI_Recv(&temp[0] + index, t, MPI_INT, i, 1, MPI_COMM_WORLD, status);
            index += t;
        }

        vec = temp;


    } else {
        int length = temp.size();
        MPI_Send(&length, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
        MPI_Send(&temp[0], length, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
    }
}





int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size bucket_num\n"
            );
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

    const int size = atoi(argv[1]);

    // I choose 10000 as optimal number of buckets
    const int bucket_num = atoi(argv[2]);

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    bucketSort(vec, bucket_num, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Bucket Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}