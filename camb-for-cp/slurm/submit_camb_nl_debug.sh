#!/bin/bash
#SBATCH --job-name=camb_nl_dbg
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:25:00
#SBATCH --account=des
#SBATCH --output=logs/camb_nl_dbg_%A_%a.out
#SBATCH --error=logs/camb_nl_dbg_%A_%a.err
#SBATCH --array=0-1

# Debug-queue smoke test for the v2 NONLINEAR CAMB pipeline.
# Two tasks, one per box (wide, dense). Both run concurrently at debug
# qos (MaxJobsPU=2 budget). Each task processes 100 cosmologies x 11
# z-slices = 1100 rows in ~8-15 min wall (halofit adds ~3x the linear
# cost). Stays well under the 25 min wall-cap with headroom.
#
# Expected output: two .dat files per box, each with 1100 rows × 513
# cols (7 leading cols: 6 params + z, then 506 P(k) values).
#
# Submit:
#   sbatch slurm/submit_camb_nl_debug.sh

export OMP_NUM_THREADS=1

# y3_cluster_cpp prescription: source setup-cosmosis-nersc directly.
export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc \
    /global/common/software/des/common/Conda_Envs/y3cl_je

cd "${SLURM_SUBMIT_DIR:-$PSCRATCH/camb-emulator/camb-for-cp}"

# -- Box selection ---------------------------------------------------------
# Task 0 -> wide, task 1 -> dense (the two boxes already produced for v3c).
# If the LHS slices for the requested box don't yet exist, regenerate
# them via create_lhs_params_1M.py + make_lhs_lists_1M.py +
# split_lhs_for_array.py (see HOWTO_1M.md steps 1-3).
BOX_LIST=(wide dense)
BOX=${BOX_LIST[${SLURM_ARRAY_TASK_ID}]}

SLICE_DIR=./slices_1M_${BOX}
SLICE_ID=dbg_nl
SRC=${SLICE_DIR}/slice_000.list
DST=${SLICE_DIR}/slice_debug_nl.list

if [ ! -f "${SRC}" ]; then
    echo "ERROR: ${SRC} not found."
    echo "Run HOWTO_1M.md steps 1-3 first to generate LHS slices for"
    echo "the ${BOX} box, then re-submit."
    exit 1
fi
head -101 "${SRC}" > "${DST}"   # 1 header + 100 data rows

export LHS_SLICE_FILE=${DST}
export SAVE_PK_PREFIX=slice_debug_nl_${BOX}_
export SLICE_ID=debug_nl_${BOX}

# Clean any prior attempt so append-mode writes start fresh.
for f in "${SAVE_PK_PREFIX}linear_nl_v2.dat" \
         "${SAVE_PK_PREFIX}linear_nonu_nl_v2.dat" \
         "${SAVE_PK_PREFIX}k_modes_v2.txt"; do
    [ -f "$f" ] && rm -f "$f"
done

echo "========== camb_nl_debug (${BOX}) =========="
echo "task id:      ${SLURM_ARRAY_TASK_ID}"
echo "slice file:   ${LHS_SLICE_FILE} (100 cosmologies)"
echo "z grid:       0.0 -> 1.0, nz=11 (halofit)"
echo "out prefix:   ${SAVE_PK_PREFIX}"
echo "expect rows:  1100 per .dat  (=100 x 11)"
echo "expect cols:  513            (=7 leading + 506 P(k))"
echo "date:         $(date)"
echo "============================================="

cosmosis configs/camb_pipeline_training_nl.ini

echo ""
echo "Finished (${BOX}): $(date)"

# Sanity check on output
LIN="${SAVE_PK_PREFIX}linear_nl_v2.dat"
NON="${SAVE_PK_PREFIX}linear_nonu_nl_v2.dat"
if [ -f "${LIN}" ]; then
    nl=$(wc -l < "${LIN}")
    nc=$(head -1 "${LIN}" | awk '{print NF}')
    echo "linear_nl rows=${nl} cols=${nc}"
else
    echo "ERROR: ${LIN} missing"
    exit 1
fi
if [ -f "${NON}" ]; then
    nn=$(wc -l < "${NON}")
    echo "linear_nonu_nl rows=${nn}"
    if [ "${nl}" = "${nn}" ] && [ "${nc}" = "513" ]; then
        echo "OK: matching rows, 513 cols"
    else
        echo "WARN: linear/nonu mismatch or cols != 513"
    fi
else
    echo "WARN: ${NON} missing -- CAMB may not emit cdm_baryon_power_nl."
    echo "      Single-network linear_nl path is still usable."
    if [ "${nc}" = "513" ]; then
        echo "OK (single-network): linear_nl cols=513"
    else
        echo "ERROR: linear_nl cols != 513 (got ${nc})"
        exit 1
    fi
fi
