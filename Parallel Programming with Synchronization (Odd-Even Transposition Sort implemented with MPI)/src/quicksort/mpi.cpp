// 120090712
//
// Parallel Quick Sort with MPI
//


#include <iostream>
#include <vector>
#include <mpi.h>
#include "../utils.hpp"

#define MASTER 0


// Min heap operations:
void heap_swap(std::vector<int> &heap, std::vector<int> &location, int index1, int index2){
    int temp1 = heap[index1];
    heap[index1] = heap[index2];
    heap[index2] = temp1;

    int temp2 = location[index1];
    location[index1] = location[index2];
    location[index2] = temp2;
}

void min_heap_up(std::vector<int> &heap, std::vector<int> &location, int index){
    while (index > 0){
        int parent = (index - 1)/2;
        if (heap[index] < heap[parent]){
            heap_swap(heap, location, index, parent);
            index = parent;
        } else {
            break;
        }
    }
}

void min_heap_down(std::vector<int> &heap, std::vector<int> &location, int index){
    int size = heap.size();
    while(true){
        int left = index * 2 + 1;
        int right = index * 2 + 2;
        int smallest = index;
        if (left < size && heap[left] < heap[smallest]){
            smallest = left;
        }
        if (right < size && heap[right] < heap[smallest]){
            smallest = right;
        }
        if (smallest != index){
            heap_swap(heap,location,index,smallest);
            index = smallest;
        } else {
            break;
        }

    }
}

void min_heap_insert(std::vector<int> &heap, std::vector<int> &location, int value, int idx){
    heap.push_back(value);
    location.push_back(idx);
    min_heap_up(heap, location, heap.size()-1);
}

void min_heap_replace(std::vector<int> &heap, std::vector<int> &location, int value, int idx){
    heap[0] = value;
    location[0] = idx;
    min_heap_down(heap, location, 0);
}


void min_heap_pop(std::vector<int> &heap, std::vector<int> &location){
    heap[0] = heap[heap.size()-1];
    location[0] = location[location.size()-1];
    heap.pop_back();
    location.pop_back();
    min_heap_down(heap, location, 0);
}


void k_merge(std::vector<int> &vec,std::vector<int> &cuts,int k){
    // info is used to check whether the k chunks are empty
    std::vector<int> result;
    std::vector<int> info(k,0);
    std::vector<int> min_heap;
    std::vector<int> location;
    // Initialize the k-min_heap
    for (int i = 0; i < k; ++i){
        if (cuts[i] + info[i] < cuts[i + 1]){
            min_heap_insert(min_heap, location, vec[cuts[i] + info[i]], i);
            info[i] = info[i] + 1;
        }
    }
    while(true){
        if (min_heap.empty()){
            break;
        }
        int smallest = min_heap[0];
        int idx = location[0];
        result.push_back(smallest);
        if (cuts[idx] + info[idx] < cuts[idx + 1]){
            // Replace the top integer
            min_heap_replace(min_heap, location, vec[cuts[idx] + info[idx]], idx);
            info[idx] = info[idx] + 1;
        } else {
            min_heap_pop(min_heap, location);
        }
    }
    vec = result;
}


int partition(std::vector<int> &vec, int low, int high) {
    int pivot = vec[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            ++i;
            std::swap(vec[i], vec[j]);
        }
    }

    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

void quickSort_for_pro(std::vector<int> &vec, int low, int high){
    if (low < high) {
        int pivotIndex = partition(vec, low, high);
        quickSort_for_pro(vec, low, pivotIndex - 1);
        quickSort_for_pro(vec, pivotIndex + 1, high);
    }    
}


void quickSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    // Implement parallel quick sort with MPI
    // There are 0 ~ size- 1 elements in the vector
    int length = vec.size();

    // Divide the works
    int numbers_per_process = length / numtasks;
    int left_numbers = length % numtasks;

    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_numbers = 0;

    for (int i = 0; i < numtasks; ++i) {
        if (divided_left_numbers < left_numbers) {
            cuts[i+1] = cuts[i] + numbers_per_process + 1;
            divided_left_numbers++;
        } else cuts[i+1] = cuts[i] + numbers_per_process;
    }

    // Do the sorting
    quickSort_for_pro(vec,cuts[taskid], cuts[taskid+1] - 1);

    if (taskid == MASTER){
        for (int i = MASTER + 1; i < numtasks; ++i) {
            int* start_pos = &vec[0] + cuts[i];
            MPI_Recv(start_pos, cuts[i+1] - cuts[i], MPI_INT, i, 0, MPI_COMM_WORLD, status);
        }
        // Mantain a min-heap
        k_merge(vec,cuts,numtasks);

    } else {
        MPI_Send(&vec[cuts[taskid]], cuts[taskid + 1] - cuts[taskid], MPI_INT, MASTER, 0, MPI_COMM_WORLD);
    }

}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
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

    const int seed = 4005;

    std::vector<int> vec = createRandomVec(size, seed);
    std::vector<int> vec_clone = vec;

    auto start_time = std::chrono::high_resolution_clock::now();

    quickSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Quick Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}