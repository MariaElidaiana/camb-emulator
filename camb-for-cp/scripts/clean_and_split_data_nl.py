#!/usr/bin/env python
"""
Convert v2-nl pre-split .dat files to .npy for the z-aware nonlinear
emulator.

Unlike clean_and_split_data_v2.py (6 params, z dropped), rows here have
a z column as the 7th leading element:
  [h0, omega_m, omega_b, n_s, log1e10As, mnu, z, P(k1), ..., P(kN)]
so PARAM_NAMES is 7-long and the emulator takes z as an input feature.

Expects train/test splits done externally (shuffled 90/10 with a shared
permutation across linear_nl and linear_nonu_nl so row i is the same
(cosmo, z) in both):
  linear_nl_v2_train.dat      linear_nl_v2_test.dat
  linear_nonu_nl_v2_train.dat linear_nonu_nl_v2_test.dat

Cleaning: drop rows with non-finite params/spectra or non-positive P(k).
"""

import gc
import os
import time

import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    import pandas as pd
    HAS_POLARS = False
    print("WARNING: polars not installed, falling back to pandas (slower)")

K_FILE = "k_modes_v2.txt"
OUTPUT_DIR = "./training_data_nl_v2"

PARAM_NAMES = ["h0", "omega_m", "omega_b", "n_s", "log1e10As", "mnu", "z"]
N_PARAMS = len(PARAM_NAMES)

FILES = [
    ("linear_nl_v2",      "train", "linear_nl_v2_train.dat"),
    ("linear_nl_v2",      "test",  "linear_nl_v2_test.dat"),
    ("linear_nonu_nl_v2", "train", "linear_nonu_nl_v2_train.dat"),
    ("linear_nonu_nl_v2", "test",  "linear_nonu_nl_v2_test.dat"),
]


def read_dat(filename):
    if HAS_POLARS:
        df = pl.read_csv(
            filename, separator=" ", has_header=False,
            dtypes=[pl.Float64], ignore_errors=True,
        )
        df = df.select([c for c in df.columns if not df[c].is_null().all()])
        return df.to_numpy()
    return pd.read_csv(filename, sep=r"\s+", header=None,
                       dtype=np.float64, engine="c").values


def convert(filename, quantity, split, n_k, k_modes):
    t0 = time.time()
    print(f"  Reading {filename}...")
    data = read_dat(filename)
    total = len(data)
    print(f"  Read {total:,} rows in {time.time() - t0:.1f}s")

    params = data[:, :N_PARAMS]
    spectra = data[:, N_PARAMS:]
    del data
    gc.collect()

    valid = (np.isfinite(spectra).all(axis=1)
             & np.isfinite(params).all(axis=1)
             & (spectra > 0).all(axis=1)
             & (spectra > 1e-12).all(axis=1))

    n_removed = (~valid).sum()
    params = params[valid].astype(np.float32)
    spectra = np.log10(spectra[valid]).astype(np.float32)

    assert spectra.shape[1] == n_k, \
        f"Spectrum cols {spectra.shape[1]} != k-modes {n_k}"

    n_valid = len(params)
    print(f"  Valid: {n_valid:,}, removed: {n_removed:,} "
          f"({100 * n_removed / total:.2f}%)")

    np.save(os.path.join(OUTPUT_DIR, f"camb_{quantity}_params_{split}.npy"), params)
    np.save(os.path.join(OUTPUT_DIR, f"camb_{quantity}_logpower_{split}.npy"), spectra)
    if split == "train":
        np.save(os.path.join(OUTPUT_DIR, f"camb_{quantity}_modes.npy"), k_modes)

    print(f"  Saved .npy for camb_{quantity}_{split} "
          f"({time.time() - t0:.1f}s total)")

    del params, spectra
    gc.collect()
    return n_valid


def main():
    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    k_modes = np.loadtxt(K_FILE)
    n_k = len(k_modes)
    print(f"Loaded {n_k} k-modes from {K_FILE}")
    print(f"Backend: {'polars' if HAS_POLARS else 'pandas'}\n")

    for quantity, split, filename in FILES:
        if not os.path.exists(filename):
            print(f"SKIP {filename} (not found)")
            continue
        out_params = os.path.join(OUTPUT_DIR,
                                  f"camb_{quantity}_params_{split}.npy")
        out_features = os.path.join(OUTPUT_DIR,
                                    f"camb_{quantity}_logpower_{split}.npy")
        if os.path.exists(out_params) and os.path.exists(out_features):
            print(f"SKIP {quantity} {split} (output already exists)")
            continue
        print(f"Converting {quantity} {split}:")
        convert(filename, quantity, split, n_k, k_modes)
        print()

    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
