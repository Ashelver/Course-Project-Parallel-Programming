#!/bin/bash
#SBATCH -o ./Project1-PartB-Results.txt
#SBATCH -p Project
#SBATCH -J Project1-PartB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Set the current directory important!!!!!!!!!!!!!!!!!!!!!
CURRENT_DIR="/nfsmnt/120090712/CSC4005-2023Fall/project1/src/scripts"
echo "Current directory: ${CURRENT_DIR}"

# Sequential PartB
echo "Sequential PartB (Optimized with -O3)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../../build/src/cpu/sequential_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth-SEQ.jpg
echo ""

# SIMD PartB
echo "SIMD(AVX2) PartB (Optimized with -O3)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../../build/src/cpu/simd_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth-SIMD.jpg
echo ""

# MPI PartB
echo "MPI PartB (Optimized with -O3)"
for num_processes in 1 2 4 8 16 32
do
  echo "Number of processes: $num_processes"
  srun -n $num_processes --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../../build/src/cpu/mpi_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth-MPI.jpg
  echo ""
done

# Pthread PartB
echo "Pthread PartB (Optimized with -O3)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../../build/src/cpu/pthread_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth-PTHREAD.jpg ${num_cores}
  echo ""
done

# OpenMP PartB
echo "OpenMP PartB (Optimized with -O3)"
for num_cores in 1 2 4 8 16 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../../build/src/cpu/openmp_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth-OMP.jpg ${num_cores}
  echo ""
done

# CUDA PartB
echo "CUDA PartB"
srun -n 1 --gpus 1 ${CURRENT_DIR}/../../build/src/gpu/cuda_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth-CUDA.jpg
echo ""

# OpenACC PartB
echo "OpenACC PartB"
srun -n 1 --gpus 1 ${CURRENT_DIR}/../../build/src/gpu/openacc_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth-OPENACC.jpg
echo ""
