#!/bin/bash
#SBATCH --job-name=camb_nl_1M
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:30:00
#SBATCH --account=des
#SBATCH --output=logs/camb_nl_1M_%A_%a.out
#SBATCH --error=logs/camb_nl_1M_%A_%a.err
#SBATCH --array=0-0%50

# 1M-sample NONLINEAR (halofit) CAMB generation on the shared qos.
# Runs *one* box at a time; choose by exporting BOX and overriding --array:
#
#   sbatch --array=0-299%50 --export=BOX=wide   slurm/submit_camb_training_nl_1M_array.sh
#   sbatch --array=0-499%50 --export=BOX=dense  slurm/submit_camb_training_nl_1M_array.sh
#   sbatch --array=0-199%50 --export=BOX=ultra  slurm/submit_camb_training_nl_1M_array.sh
#
# (Defaults above are placeholders; run_1M_nl_post_camb.sh documents the
#  correct --array per box.)
#
# Each task processes 1000 samples x 11 z-slices = 11000 rows serially via
# cosmosis (no MPI). Halofit at nz=11 makes each sample ~1.5x the linear
# cost, so wall is 4.5 h (vs 2.5 h for the linear 1M run).
#
# Output files are prefixed 'slice{NNN}{box}_' so boxes don't collide and
# concurrent tasks don't touch the same inode:
#   slice{NNN}{box}_linear_nl_v2.dat
#   slice{NNN}{box}_linear_nonu_nl_v2.dat
#   slice{NNN}{box}_k_modes_v2.txt

export OMP_NUM_THREADS=1

# y3_cluster_cpp prescription: source setup-cosmosis-nersc directly.
export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc \
    /global/common/software/des/common/Conda_Envs/y3cl_je

#cd "${SLURM_SUBMIT_DIR:-$PSCRATCH/camb-emulator/camb-for-cp}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")}"

# -- Box selection --------------------------------------------------------
BOX=${BOX:-wide}
case "${BOX}" in
    wide|dense|ultra) ;;
    *)
        echo "ERROR: BOX must be 'wide', 'dense', or 'ultra' (got '${BOX}')"
        exit 1
        ;;
esac
SLICE_DIR=./slices_1M_${BOX}
SLICE_ID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
export SLICE_ID
export LHS_SLICE_FILE=${SLICE_DIR}/slice_${SLICE_ID}.list
export SAVE_PK_PREFIX=slice${SLICE_ID}${BOX}_

# -- Re-run safety --------------------------------------------------------
# If a previous (preempted) attempt left partial output, clean it so
# append-mode writes start fresh. Idempotent per slice.
for f in "${SAVE_PK_PREFIX}linear_nl_v2.dat" \
         "${SAVE_PK_PREFIX}linear_nonu_nl_v2.dat" \
         "${SAVE_PK_PREFIX}k_modes_v2.txt"; do
    [ -f "$f" ] && rm -f "$f"
done

echo "========== camb_nl_1M array task =========="
echo "Box:          ${BOX}"
echo "Array:        ${SLURM_ARRAY_JOB_ID}, task: ${SLURM_ARRAY_TASK_ID}"
echo "Slice file:   ${LHS_SLICE_FILE}"
echo "Output prefix: ${SAVE_PK_PREFIX}"
echo "Expect rows:  11000 per .dat (=1000 x 11 z-slices)"
echo "Node:         $(hostname)"
echo "Date:         $(date)"
echo "========================================="

# Nonlinear pipeline config (nonlinear=pk, halofit takahashi, zmax=1, nz=11).
cosmosis configs/camb_pipeline_training_nl.ini

echo "Finished task ${SLURM_ARRAY_TASK_ID} (${BOX}): $(date)"