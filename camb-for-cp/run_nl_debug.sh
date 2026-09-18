#!/bin/bash
# ---------------------------------------------------------------------------
# Driver for the nonlinear-emulator debug smoke test.
#
# Run one step at a time and inspect outputs before the next. The 30-min
# debug qos is a binding constraint -- each step is short-enough to fit
# there.
#
# Prerequisite: LHS slices for the `wide` and `dense` boxes already
# exist under ./slices_1M_{wide,dense}/slice_000.list (these are the
# v3c artifacts -- see HOWTO_1M.md steps 1-3 to regenerate if missing,
# then re-run the `merge` and `split` steps below).
#
# Usage:
#   ./run_nl_debug.sh emit     # 1. SLURM array: 100 cosmologies x 11 z-slices per box
#   ./run_nl_debug.sh merge    # 2. concat the 2 per-box .dat files per quantity
#   ./run_nl_debug.sh split    # 3. shuffled 90/10 train/test (shared perm)
#   ./run_nl_debug.sh clean    # 4. SLURM job: .dat -> .npy under training_data_nl_v2/
#   ./run_nl_debug.sh train    # 5. SLURM jobs (2): GPU training, shortened schedule
#   ./run_nl_debug.sh compare  # 6. (manual) open evaluate_emulator.ipynb
# ---------------------------------------------------------------------------
set -eu

REPO="$(dirname "$(realpath "$0")")"
cd "$REPO"

PYBIN_TRAIN=/global/common/software/des/common/Conda_Envs/jesteves_cosmopower/bin/python

STEP=${1:-}

case "$STEP" in

emit)
    echo "Submitting nonlinear CAMB debug array (2 tasks: wide, dense)..."
    sbatch slurm/submit_camb_nl_debug.sh
    echo ""
    echo "Wait for both tasks to COMPLETE, then verify outputs before"
    echo "running the next step:"
    echo "    for box in wide dense; do"
    echo "        f=slice_debug_nl_\${box}_linear_nl_v2.dat"
    echo "        [ -f \"\$f\" ] && echo \"\$box: rows=\$(wc -l < \$f) cols=\$(head -1 \$f | awk '{print NF}')\""
    echo "    done"
    echo "Expected: rows=1100 cols=513 for each box."
    echo "Then run: ./run_nl_debug.sh merge"
    ;;

merge)
    # Concatenate the 2 per-box .dat files per quantity into a single
    # .dat per quantity. (Wide + dense = 200 cosmologies x 11 z = 2200
    # training rows total.)
    echo "Concatenating per-box debug outputs..."
    for name in linear_nl_v2 linear_nonu_nl_v2; do
        inputs=""
        for box in wide dense; do
            f="slice_debug_nl_${box}_${name}.dat"
            if [ -f "$f" ]; then
                inputs="${inputs} $f"
            else
                echo "WARN: $f not found -- skipping this box"
            fi
        done
        if [ -z "${inputs}" ]; then
            echo "ERROR: no per-box ${name} files found. Run 'emit' first."
            exit 1
        fi
        eval "cat ${inputs} > ${name}.dat"
        echo "  -> ${name}.dat: $(wc -l < ${name}.dat) rows"
    done

    # k_modes_v2.txt is written by the emit step per-box (prefixed as
    # slice_debug_nl_${box}_k_modes_v2.txt). The clean step's cleaner
    # expects the unprefixed name at the repo root. Both boxes should
    # emit byte-identical k-grids (same CAMB kmin/kmax/nk); cross-check
    # with md5sum and copy the wide one to the canonical name.
    K_WIDE="slice_debug_nl_wide_k_modes_v2.txt"
    K_DENSE="slice_debug_nl_dense_k_modes_v2.txt"
    if [ -f "${K_WIDE}" ] && [ -f "${K_DENSE}" ]; then
        m_w=$(md5sum "${K_WIDE}"   | awk '{print $1}')
        m_d=$(md5sum "${K_DENSE}" | awk '{print $1}')
        if [ "${m_w}" != "${m_d}" ]; then
            echo "WARN: k-modes files differ between boxes"
            echo "  wide:  ${m_w}"
            echo "  dense: ${m_d}"
            echo "  Using wide."
        fi
        cp "${K_WIDE}" k_modes_v2.txt
        echo "  -> k_modes_v2.txt: $(wc -l < k_modes_v2.txt) modes"
    elif [ -f "${K_WIDE}" ]; then
        cp "${K_WIDE}" k_modes_v2.txt
        echo "  -> k_modes_v2.txt (from wide): $(wc -l < k_modes_v2.txt) modes"
    elif [ -f "${K_DENSE}" ]; then
        cp "${K_DENSE}" k_modes_v2.txt
        echo "  -> k_modes_v2.txt (from dense): $(wc -l < k_modes_v2.txt) modes"
    else
        echo "ERROR: no k_modes_v2.txt found in either box output."
        echo "Rerun: ./run_nl_debug.sh emit"
        exit 1
    fi

    # Integrity check: linear and nonu must match row-for-row.
    n_lin=$(wc -l < linear_nl_v2.dat)
    if [ -f linear_nonu_nl_v2.dat ]; then
        n_nonu=$(wc -l < linear_nonu_nl_v2.dat)
        if [ "$n_lin" != "$n_nonu" ]; then
            echo "ERROR: linear_nl rows ${n_lin} != linear_nonu_nl rows ${n_nonu}"
            echo "Truncate the longer file or rerun 'emit' for the missing box."
            exit 1
        fi
        echo "OK: linear/nonu row counts match (${n_lin})"
    else
        echo "WARN: linear_nonu_nl_v2.dat not produced (CAMB may not emit"
        echo "      cdm_baryon_power_nl with current interface version)."
        echo "      Continuing with single-network linear_nl_v2 only."
    fi
    ;;

split)
    # Shuffled 90/10 train/test split. Shared permutation across linear
    # and nonu so row i is the same (cosmo, z) in both.
    echo "Shuffling + splitting 90/10..."
    ${PYBIN_TRAIN} - <<'PY'
import os
import numpy as np
import polars as pl

# Read both (or just linear_nl if nonu absent).
dfs = {}
for name in ('linear_nl_v2', 'linear_nonu_nl_v2'):
    f = f'{name}.dat'
    if not os.path.exists(f):
        print(f'SKIP {f} (not found)')
        continue
    df = pl.read_csv(f, separator=' ', has_header=False,
                     schema_overrides=[pl.Float64], ignore_errors=True)
    dfs[name] = df.select([c for c in df.columns if not df[c].is_null().all()]).to_numpy()

A = dfs['linear_nl_v2']
print(f'linear_nl_v2 rows: {len(A)}')
if 'linear_nonu_nl_v2' in dfs:
    B = dfs['linear_nonu_nl_v2']
    assert A.shape == B.shape, f'{A.shape} vs {B.shape}'
    print(f'linear_nonu_nl_v2 rows: {len(B)} (shapes match)')
else:
    B = None

rng = np.random.default_rng(20260507)
perm = rng.permutation(len(A))
A = A[perm]
if B is not None:
    B = B[perm]

n_train = int(len(A) * 0.9)
np.savetxt('linear_nl_v2_train.dat',      A[:n_train], fmt='%.8e')
np.savetxt('linear_nl_v2_test.dat',       A[n_train:], fmt='%.8e')
print(f'linear_nl_v2: train={n_train} test={len(A)-n_train}')

if B is not None:
    np.savetxt('linear_nonu_nl_v2_train.dat', B[:n_train], fmt='%.8e')
    np.savetxt('linear_nonu_nl_v2_test.dat',  B[n_train:], fmt='%.8e')
    print(f'linear_nonu_nl_v2: train={n_train} test={len(B)-n_train}')
PY
    echo "Split OK."
    ;;

clean)
    echo "Submitting clean_split on debug qos..."
    sbatch slurm/submit_clean_split_nl_debug.sh
    echo "Wait for COMPLETED, then check training_data_nl_v2/*.npy"
    ;;

train)
    echo "Submitting two GPU training jobs (shortened schedule for debug)..."
    sbatch --export=SPECTRA=linear_nl_v2      slurm/submit_train_nl_debug.sh
    if [ -f linear_nonu_nl_v2_train.dat ]; then
        sbatch --export=SPECTRA=linear_nonu_nl_v2 slurm/submit_train_nl_debug.sh
    else
        echo "(linear_nonu_nl_v2_train.dat not found -- skipping 2nd job)"
    fi
    echo "Wait for jobs to COMPLETE; models written as camb_*_nl_v2_emulator.pkl"
    ;;

compare)
    echo "Manual step: open scripts/evaluate_emulator.ipynb and"
    echo "run the new NL comparison cell at the bottom of the notebook."
    echo "It loads camb_linear_nl_v2_emulator and a held-out cosmology"
    echo "and plots P(k) at z in {0, 0.5, 1.0} vs cosmosis+CAMB halofit."
    ;;

*)
    echo "Usage: $0 <emit|merge|split|clean|train|compare>"
    exit 1
    ;;

esac
