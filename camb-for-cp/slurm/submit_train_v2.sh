#!/bin/bash
#SBATCH --job-name=train_v2
#SBATCH --qos=regular
#SBATCH -C gpu&hbm80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --account=des_g
#SBATCH --output=logs/train_v2_%j.out
#SBATCH --error=logs/train_v2_%j.err

# Trains ONE emulator. Pass the spectra name via --export
# e.g. sbatch --export=SPECTRA=linear_v2 slurm/submit_train_v2.sh
# or   sbatch --export=SPECTRA=linear_nl_v3c slurm/submit_train_v2.sh
# Defaults to linear_v2 if SPECTRA is unset.
# Wall is 12 h to cover the 11M-row nonlinear (LARGE batch) schedule;
# the trainer auto-resumes from camb_{spectra}_emulator.pkl, so if it
# times out, resubmitting the same job continues from the checkpoint.

#cd "${SLURM_SUBMIT_DIR:-$PSCRATCH/camb-emulator/camb-for-cp}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")}"

module load tensorflow/2.15.0

SPECTRA=${SPECTRA:-linear_v2}

echo "=== train_emulator_v2 (spectra=${SPECTRA}) ==="
date
echo "GPUs: $SLURM_GPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import tensorflow as tf; print('TF GPUs:', tf.config.list_physical_devices('GPU'))"

python scripts/train_emulator_v2.py --spectra ${SPECTRA}

date
echo "Finished"
