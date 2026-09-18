#!/usr/bin/env python
"""
Shuffled 90/10 train/test split for the 1M NONLINEAR (v3c-nl) .dat files.

The merged linear_nl_v3c.dat / linear_nonu_nl_v3c.dat are ~85 GB each
(11M rows x 513 cols). Holding both as numpy arrays plus a permutation
would need ~90+ GB of RAM (float64) -- this streams both files row-by-row
instead, writing each line to the train or test output based on a boolean
mask derived from a shared permutation.

Memory footprint: 1 bool per row (~11 MB) + a line buffer. No float parsing.

The assignment (which rows are train/test) is identical across both files
because it uses one permutation. Row ORDER inside the outputs is not the
shuffled order -- it is the thinned original order -- which is fine: the
trainer's --nsamples path (train_emulator_v2.py) does its own
np.random.choice before training, and CosmoPower shuffles before each epoch.

Usage:
    python split_nl_v3c.py [--frac-train 0.9] [--seed 20260507]

Inputs:   linear_nl_v3c.dat, linear_nonu_nl_v3c.dat
Outputs:  {linear_nl_v3c,linear_nonu_nl_v3c}_{train,test}.dat
"""

import argparse
import os
import sys
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--frac-train", type=float, default=0.9)
parser.add_argument("--seed", type=int, default=20260507)
parser.add_argument("--names", nargs="+",
                    default=["linear_nl_v3c", "linear_nonu_nl_v3c"])
args = parser.parse_args()


def stream_split(filename, is_train):
    """Write rows of `filename` to <name>_train.dat/_test.dat by line.

    Returns (n_rows, n_train, n_test). Passes the line through verbatim
    (no float reformatting) so the output .dat matches the input byte style.
    """
    stem = filename[:-4] if filename.endswith(".dat") else filename
    n_rows = 0
    n_tr = 0
    n_te = 0
    out_tr = open(f"{stem}_train.dat", "wb", buffering=1 << 20)
    out_te = open(f"{stem}_test.dat", "wb", buffering=1 << 20)
    try:
        with open(filename, "rb") as f:
            for raw in f:
                if not raw.strip():
                    n_rows += 1
                    continue
                if is_train[n_rows]:
                    out_tr.write(raw)
                    n_tr += 1
                else:
                    out_te.write(raw)
                    n_te += 1
                n_rows += 1
    finally:
        out_tr.close()
        out_te.close()
    return n_rows, n_tr, n_te


def main():
    t0 = time.time()

    for name in args.names:
        if not os.path.exists(f"{name}.dat"):
            print(f"ERROR: {name}.dat missing (run merge first)",
                  file=sys.stderr)
            sys.exit(1)

    # First pass: count rows on the FIRST file (counts must match across
    # files -- the merge step already asserted this).
    first = args.names[0]
    print(f"Counting rows in {first}.dat ...")
    with open(f"{first}.dat", "rb") as f:
        n_rows = sum(1 for _ in f)
    print(f"  {n_rows:,} rows")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_rows)
    n_train = int(n_rows * args.frac_train)
    is_train = np.zeros(n_rows, dtype=bool)
    is_train[perm[:n_train]] = True
    del perm

    for name in args.names:
        print(f"Splitting {name}.dat  (train={n_train:,}, "
              f"test={n_rows - n_train:,}) ...", flush=True)
        nr, ntr, nte = stream_split(f"{name}.dat", is_train)
        if nr != n_rows:
            print(f"ERROR: {name}.dat row count {nr} != {n_rows}",
                  file=sys.stderr)
            sys.exit(1)
        # Sanity check: train/test counts match expected split exactly.
        if ntr != n_train or nte != n_rows - n_train:
            print(f"ERROR: {name} split produced train={ntr} test={nte}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"  -> {name}_train.dat: {ntr:,} | {name}_test.dat: {nte:,}")

    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
