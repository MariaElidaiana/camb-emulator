#!/bin/bash
#SBATCH --job-name=clean_nl
#SBATCH --qos=shared
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --account=des
#SBATCH --output=logs/clean_nl_%j.out
#SBATCH --error=logs/clean_nl_%j.err

# Production .dat -> .npy conversion for the 1M NONLINEAR (v3c-nl) data.
# The merged train/test .dat files are ~85 GB each (11M rows x 513 cols), so
# this MUST run on a node with enough RAM and MUST use the chunked reader in
# scripts/clean_and_split_data_nl_v3c.py (streaming, bounded memory).
#
# A login node is NOT acceptable: the clean step's peak memory (streaming
# parse + float32 concat) is ~20-40 GB.
#
# Submit:
#   sbatch slurm/submit_clean_split_nl.sh
# or via the driver:  ./run_1M_nl_post_camb.sh clean

#cd "${SLURM_SUBMIT_DIR:-$PSCRATCH/camb-emulator/camb-for-cp}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")}"

PYBIN=/global/common/software/des/common/Conda_Envs/jesteves_cosmopower/bin/python
echo "Using python: ${PYBIN}"
${PYBIN} --version

echo "=== clean_and_split_data_nl_v3c (chunked) ==="
date

${PYBIN} scripts/clean_and_split_data_nl_v3c.py

date
echo "Finished"