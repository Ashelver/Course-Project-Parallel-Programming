// 120090712
//
// Parallel Odd-Even Sort with MPI
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

void send_value(int dest, int value){
    MPI_Request request;
    // MPI_Isend(&value, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, &request);
    MPI_Send(&value, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
}

int receive_value(int sourc){
    MPI_Request request;
    int value;
    // MPI_Irecv(&value, 1, MPI_INT, sourc, 0, MPI_COMM_WORLD, &request);
    MPI_Recv(&value, 1, MPI_INT, sourc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return value;
}

int even_phase_for_odd(int taskid, bool flag, std::vector<int>&fragment, int sorted){
    if (fragment.size() == 0) return sorted;
    flag &= (fragment.size()&1);
    if (flag) fragment.push_back(receive_value(taskid + 1));
    for (int i = 0; i < fragment.size() - 1; i += 2) {
        if (fragment[i] > fragment[i + 1]) {
            std::swap(fragment[i], fragment[i + 1]);
            sorted = 0;
        }
    }
    if (flag) {
        send_value(taskid + 1, fragment[fragment.size() - 1]);
        fragment.pop_back();
    }
    return sorted;
}


int even_phase_for_even(int taskid, bool flag, std::vector<int>&fragment, int sorted){
    if (fragment.size() == 0) return sorted;
    flag &= (fragment.size()&1);
    if (flag) fragment.push_back(receive_value(taskid + 1));
    for (int i = 0; i < fragment.size() - 1; i += 2) {
        if (fragment[i] > fragment[i + 1]) {
            std::swap(fragment[i], fragment[i + 1]);
            sorted = 0;
        }
    }
    if (flag) {
        send_value(taskid + 1, fragment[fragment.size() - 1]);
        fragment.pop_back();
    }
    return sorted;
}


int odd_phase_for_odd(int taskid, bool flag, std::vector<int>&fragment, int sorted){
    if (fragment.size() == 0) return sorted;
    flag &= (!(fragment.size()&1));
    if (taskid !=0) send_value(taskid - 1, fragment[0]);
    if (flag) fragment.push_back(receive_value(taskid + 1));
    for (int i = 1; i < fragment.size() - 1; i += 2) {
        if (fragment[i] > fragment[i + 1]) {
            std::swap(fragment[i], fragment[i + 1]);
            sorted = 0;
        }
    }
    if (taskid !=0) fragment[0] = receive_value(taskid - 1);
    if (flag) {
        send_value(taskid + 1, fragment[fragment.size() - 1]);
        fragment.pop_back();
    }
    return sorted;    
}

int odd_phase_for_even(int taskid, bool flag, std::vector<int>&fragment, int sorted){
    if (fragment.size() == 0) return sorted;
    flag &= (!(fragment.size()&1));
    if (flag) fragment.push_back(receive_value(taskid + 1));
    if (taskid != 0) send_value(taskid - 1, fragment[0]);
    for (int i = 1; i < fragment.size() - 1; i += 2) {
        if (fragment[i] > fragment[i + 1]) {
            std::swap(fragment[i], fragment[i + 1]);
            sorted = 0;
        }
    }
    if (flag) {
        send_value(taskid + 1, fragment[fragment.size() - 1]);
        fragment.pop_back();
    }
    if (taskid != 0) fragment[0] = receive_value(taskid - 1);
    return sorted;    
}


void oddEvenSort(std::vector<int>& vec, int numtasks, int taskid, MPI_Status* status) {
    // Implement parallel oddEven sort with MPI
    // There are 0 ~ size- 1 elements in the vector
    int length = vec.size();

    // Divide the works
    int numbers_per_process = length / numtasks;
    int left_numbers = length % numtasks;

    int size_of;
    if (numbers_per_process == 0){
        size_of = left_numbers;
    } else {
        size_of = numtasks;
    }
    std::vector<int> cuts(numtasks + 1, 0);
    int divided_left_numbers = 0;

    for (int i = 0; i < numtasks; i++) {
        if (divided_left_numbers < left_numbers) {
            cuts[i+1] = cuts[i] + numbers_per_process + 1;
            divided_left_numbers++;
        } else cuts[i+1] = cuts[i] + numbers_per_process;
    }

    std::vector<int> fragment(vec.begin() + cuts[taskid], vec.begin() + cuts[taskid+1]);
    // show(fragment,0,fragment.size());
    int sorted = 0;
    int start = cuts[taskid];
    bool flag = (taskid != numtasks - 1);
    while (!sorted) {
        sorted = 1;
        if (start&1){
            // Do odd first
            if (taskid&1){
                // Perform the odd phase
                sorted = odd_phase_for_odd(taskid, flag, fragment, sorted);
                // Perform the even phase
                sorted = even_phase_for_odd(taskid, flag, fragment, sorted);
            } else {
                // Perform the odd phase
                sorted = odd_phase_for_even(taskid, flag, fragment, sorted);
                // Perform the even phase
                sorted = even_phase_for_even(taskid, flag, fragment, sorted);
            }
        } else {
            // Do even first
            if (taskid&1){
                // Perform the even phase
                sorted = even_phase_for_odd(taskid, flag, fragment, sorted);
                // Perform the odd phase
                sorted = odd_phase_for_odd(taskid, flag, fragment, sorted);
            } else {
                // Perform the even phase
                sorted = even_phase_for_even(taskid, flag, fragment, sorted);
                // Perform the odd phase
                sorted = odd_phase_for_even(taskid, flag, fragment, sorted);
            }
        }
        
        if (taskid == MASTER){
            for (int i = 1; i < numtasks; ++i){
                int temp = 1;
                MPI_Recv(&temp, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, status);
                sorted &= temp;
            }
            // show(fragment,0,fragment.size());
            for (int i = 1; i < numtasks; ++i){
                MPI_Send(&sorted, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        } else {
            // show(fragment,0,fragment.size());
            MPI_Send(&sorted, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            MPI_Recv(&sorted, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, status);
            // show(fragment,0,fragment.size());
            // std::cout << std::endl;
        }
    }

    // Get the result
    if (taskid == MASTER){
        // Recieve the fragments
        fragment.resize(vec.size());
        for (int i = MASTER + 1; i < numtasks; ++i){
            MPI_Recv(&fragment[0] + cuts[i], cuts[i+1] - cuts[i], MPI_INT, i, 0, MPI_COMM_WORLD, status);
        }
        // show(fragment,0,fragment.size());
        vec = fragment;
    } else {
        // TO DO
        MPI_Send(&fragment[0], cuts[taskid + 1] - cuts[taskid], MPI_INT, MASTER, 0, MPI_COMM_WORLD);
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

    oddEvenSort(vec, numtasks, taskid, &status);

    if (taskid == MASTER) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        std::cout << "Odd-Even Sort Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
                << std::endl;
        
        checkSortResult(vec_clone, vec);
    }

    MPI_Finalize();
    return 0;
}