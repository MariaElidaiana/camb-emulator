#!/bin/bash
#SBATCH --job-name=train_nl_dbg
#SBATCH --qos=debug
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:29:00
#SBATCH --account=des_g
#SBATCH --output=logs/train_nl_dbg_%j.out
#SBATCH --error=logs/train_nl_dbg_%j.err

# Debug-queue training of the v2 NONLINEAR emulator.
#
# Single GPU (A100-80GB) only -- CosmoPower's optimizer is incompatible
# with tf.distribute.MirroredStrategy (see AGENTS.md). Uses the SHORTENED
# --max-epochs schedule because the default 400/800/1200 epoch schedule
# overflows the 29 min debug wall on small datasets.
#
# Prerequisite: training_data_nl_v2/ populated by submit_clean_split_nl_debug.sh.
#
# Pass SPECTRA via --export, e.g.
#   sbatch --export=SPECTRA=linear_nl_v2      slurm/submit_train_nl_debug.sh
#   sbatch --export=SPECTRA=linear_nonu_nl_v2 slurm/submit_train_nl_debug.sh

cd "${SLURM_SUBMIT_DIR:-$PSCRATCH/camb-emulator/camb-for-cp}"

module load tensorflow/2.15.0

SPECTRA=${SPECTRA:-linear_nl_v2}

echo "=== train_emulator_v2 (spectra=${SPECTRA}) [nl debug] ==="
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python --version
python -c "import tensorflow as tf; print('TF GPUs:', tf.config.list_physical_devices('GPU'))"
python -c "import cosmopower; print('cosmopower OK')" || {
    echo ""
    echo "ERROR: cosmopower missing in this TF module env."
    echo "Run once (interactively, not in SLURM):"
    echo "    module load tensorflow/2.15.0"
    echo "    pip install --user cosmopower polars"
    echo "then resubmit this job."
    exit 1
}

# Delete any pre-existing checkpoint so a fresh run starts from scratch.
PKL="camb_${SPECTRA}_emulator.pkl"
if [ -f "${PKL}" ]; then
    echo "Removing pre-existing checkpoint: ${PKL}"
    rm -f "${PKL}"
fi

# Shortened schedule (50/100/150 epochs) fits in the 29 min debug cap
# for the ~2200-sample debug dataset (200 cosmologies x 11 z-slices).
# Skipping this flag will overrun wall time.
python scripts/train_emulator_v2.py \
    --spectra ${SPECTRA} \
    --max-epochs 50,100,150

date
echo "Finished"
