#!/bin/bash

#SBATCH -A des
#SBATCH --job-name=class_mpi
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --constraint=cpu
#SBATCH --nodes=4                # 4 CPU node
#SBATCH --ntasks-per-node=4      # 4 MPI ranks per node
#SBATCH --cpus-per-task=64       # 4 × 64 = 256 cores per node
#SBATCH --output=logs/class_mpi_%j.out
#SBATCH --error=logs/class_mpi_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=yourmail@mail.com

# Ensure log directory exists
mkdir -p logs

# Limit OpenMP threads to avoid oversubscription
export OMP_NUM_THREADS=1

# Optional: bind CPUs for better cache locality
export SLURM_CPU_BIND="cores"

# Load CosmoPower enviromnet
source /global/homes/m/mariaeli/cosmopower_ini.sh

echo "Running CLASS with MPI on $SLURM_NTASKS ranks across $SLURM_NNODES nodes"
srun python create_training_spectra_mpi.py
