#!/bin/bash
# ---------------------------------------------------------------------------
# Post-CAMB steps for the 1M NONLINEAR (v3c-nl) pipeline:
# merge -> shuffled split -> .npy cache -> 2x GPU training -> report.
#
# Run AFTER all three CAMB arrays (submit_camb_training_nl_1M_array.sh with
# BOX=wide|dense|ultra) show COMPLETED, and the per-slice integrity check is
# clean. Do NOT run end-to-end; invoke one step at a time and inspect.
#
# Usage (in order):
#   ./run_1M_nl_post_camb.sh merge     # concat slices -> linear_nl_v3c.dat
#   ./run_1M_nl_post_camb.sh split     # streaming 90/10 train/test (same perm)
#   ./run_1M_nl_post_camb.sh clean     # SLURM: .dat -> .npy (chunked)
#   ./run_1M_nl_post_camb.sh train     # SLURM: 2 A100 jobs, 12 h each
#   ./run_1M_nl_post_camb.sh report    # regenerate notebook plots
# ---------------------------------------------------------------------------
set -eu

REPO="$(dirname "$(realpath "$0")")"
cd "$REPO"

PYBIN_CAMB=/global/common/software/des/common/Conda_Envs/y3cl_je/bin/python
PYBIN_TRAIN=/global/common/software/des/common/Conda_Envs/jesteves_cosmopower/bin/python

STEP=${1:-}

case "$STEP" in

merge)
    # Streaming concat (~10-30 min depending on Lustre). ~170 GB across both
    # quantities (11M rows x 513 cols each). Per-slice .dat are kept until
    # you manually delete them.
    echo "[1/3] Merging wide..."
    ${PYBIN_CAMB} scripts/merge_pk_outputs_parallel.py \
        --glob-template 'slice*wide_{name}.dat' \
        --output-template '{name}_1M_wide.dat' \
        --names linear_nl_v2 linear_nonu_nl_v2

    echo "[2/3] Merging dense..."
    ${PYBIN_CAMB} scripts/merge_pk_outputs_parallel.py \
        --glob-template 'slice*dense_{name}.dat' \
        --output-template '{name}_1M_dense.dat' \
        --names linear_nl_v2 linear_nonu_nl_v2

    echo "[3/3] Merging ultra..."
    ${PYBIN_CAMB} scripts/merge_pk_outputs_parallel.py \
        --glob-template 'slice*ultra_{name}.dat' \
        --output-template '{name}_1M_ultra.dat' \
        --names linear_nl_v2 linear_nonu_nl_v2

    echo "Concatenating boxes into v3c-nl..."
    cat linear_nl_v2_1M_wide.dat      linear_nl_v2_1M_dense.dat      linear_nl_v2_1M_ultra.dat      > linear_nl_v3c.dat
    cat linear_nonu_nl_v2_1M_wide.dat linear_nonu_nl_v2_1M_dense.dat linear_nonu_nl_v2_1M_ultra.dat > linear_nonu_nl_v3c.dat

    # Sanity: row counts must match across the two quantities.
    n_lin=$(wc -l < linear_nl_v3c.dat)
    n_nonu=$(wc -l < linear_nonu_nl_v3c.dat)
    echo ""
    echo "linear_nl_v3c rows:      $n_lin"
    echo "linear_nonu_nl_v3c rows: $n_nonu"
    if [ "$n_lin" != "$n_nonu" ]; then
        echo "ERROR: merged row counts differ"
        exit 1
    fi
    echo "Merge OK."
    ;;

split)
    # Streaming 90/10 split with a shared permutation across the two
    # quantities. Runs on the login node (memory-bounded): ~10-20 min for
    # ~170 GB total. One pass; rows are written to train or test output in
    # original order (trainer reshuffles internally, so order is fine).
    echo "Streaming-shuffling + splitting 90/10..."
    ${PYBIN_TRAIN} scripts/split_nl_v3c.py
    echo "Split OK."
    ;;

clean)
    # .dat -> .npy on a high-mem node (chunked reader, ~30-60 min).
    echo "Submitting clean_split on shared qos ..."
    sbatch slurm/submit_clean_split_nl.sh
    echo "Wait for COMPLETED, then verify training_data_nl_v3c/ exists."
    ;;

train)
    # Two A100 jobs (regular qos, 12 h each = safe headroom over the 8-16 h
    # worst-case estimate). Auto-resume via camb_{spectra}_emulator.pkl.
    echo "Submitting two GPU training jobs (regular qos)..."
    sbatch --export=SPECTRA=linear_nl_v3c      slurm/submit_train_v2.sh
    sbatch --export=SPECTRA=linear_nonu_nl_v3c slurm/submit_train_v2.sh
    echo "Wait for both COMPLETED; models write to camb_{linear,linear_nonu}_nl_v3c_emulator.pkl"
    ;;

report)
    # Generates report/plots/{err_cdf,err_vs_k,examples}.pdf and prints the
    # full-test accuracy table to stdout; those numbers are already filled
    # into report/emulator_nl_report.tex.
    ${PYBIN_TRAIN} scripts/make_report_plots_nl.py
    cd report
    # TeX may need a module on Perlmutter login nodes; load best-effort.
    module load texlive 2>/dev/null || true
    if ! command -v pdflatex >/dev/null 2>&1; then
        echo "WARNING: pdflatex not found; skipping PDF build."
        echo "  On Perlmutter try: module load texlive"
        echo "  then rerun 'cd report && pdflatex emulator_nl_report.tex'"
        exit 0
    fi
    pdflatex emulator_nl_report.tex
    pdflatex emulator_nl_report.tex
    echo "Report at report/emulator_nl_report.pdf"
    ;;

*)
    echo "Usage: $0 <merge|split|clean|train|report>"
    exit 1
    ;;

esac