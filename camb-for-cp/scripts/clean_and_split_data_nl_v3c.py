#!/usr/bin/env python
"""
Chunked .dat -> .npy conversion for the 1M NONLINEAR emulator (v3c-nl).

Unlike the v2c path, the nonlinear dataset has ~11M rows (1M cosmologies
x 11 z-slices), so a full-materialization read (polars read_csv) would need
~45 GB of float64 RAM for the train split alone. This script streams the
text .dat line-by-line and parses with numpy, keeping memory bounded at a
few hundred MB regardless of file size.

Row layout (7 leading cols, then 506 P(k) values):
  [h0, omega_m, omega_b, n_s, log1e10As, mnu, z, P(k1), ..., P(kN)]

Inputs (produced by run_1M_nl_post_camb.sh split):
  linear_nl_v3c_train.dat      linear_nl_v3c_test.dat
  linear_nonu_nl_v3c_train.dat linear_nonu_nl_v3c_test.dat
  k_modes_v2.txt

Outputs (under ./training_data_nl_v3c/):
  camb_linear_nl_v3c_{params,logpower}_{train,test}.npy
  camb_linear_nonu_nl_v3c_{params,logpower}_{train,test}.npy
  camb_linear_nl_v3c_modes.npy   (also the _nonu_ variant)

Cleaning: drop rows with non-finite params/spectra or non-positive P(k).
"""

import gc
import os
import time

import numpy as np

K_FILE = "k_modes_v2.txt"
OUTPUT_DIR = "./training_data_nl_v3c"

PARAM_NAMES = ["h0", "omega_m", "omega_b", "n_s", "log1e10As", "mnu", "z"]
N_PARAMS = len(PARAM_NAMES)

FILES = [
    ("linear_nl_v3c",      "train", "linear_nl_v3c_train.dat"),
    ("linear_nl_v3c",      "test",  "linear_nl_v3c_test.dat"),
    ("linear_nonu_nl_v3c", "train", "linear_nonu_nl_v3c_train.dat"),
    ("linear_nonu_nl_v3c", "test",  "linear_nonu_nl_v3c_test.dat"),
]

CHUNK_ROWS = 200_000


def _parse_line(line, ncols):
    """Parse one space-separated scientific-notation row into float64."""
    s = line.strip()
    if not s:
        return None
    # np.fromstring is fast but deprecated/removed in numpy>=2.0; fall back
    # to a split-based parse when unavailable.
    try:
        arr = np.fromstring(s, sep=" ", count=ncols)
    except (AttributeError, ValueError):
        arr = np.array(s.split(), dtype=np.float64)
    if arr.size != ncols:
        return None
    return arr


def stream_data(filename, ncols, chunk_rows=CHUNK_ROWS):
    """Yield (n_rows, ncols) float64 arrays read in bounded-memory chunks."""
    buf = []
    with open(filename, "rb") as f:
        for raw in f:
            arr = _parse_line(raw.decode("ascii", "ignore"), ncols)
            if arr is None:
                continue
            buf.append(arr)
            if len(buf) >= chunk_rows:
                yield np.vstack(buf)
                buf = []
        if buf:
            yield np.vstack(buf)


def convert(filename, quantity, split, n_k, k_modes):
    t0 = time.time()
    print(f"  Converting {filename} (quantity={quantity}, {split})...")

    params_chunks = []
    spectra_chunks = []
    total = 0
    n_removed = 0

    ncols = N_PARAMS + n_k
    for chunk in stream_data(filename, ncols):
        total += chunk.shape[0]
        params = chunk[:, :N_PARAMS]
        spectra = chunk[:, N_PARAMS:]
        valid = (np.isfinite(spectra).all(axis=1)
                 & np.isfinite(params).all(axis=1)
                 & (spectra > 0).all(axis=1)
                 & (spectra > 1e-12).all(axis=1))
        n_removed += int((~valid).sum())
        params_chunks.append(params[valid].astype(np.float32))
        spectra_chunks.append(np.log10(spectra[valid]).astype(np.float32))

    params = np.concatenate(params_chunks) if params_chunks else np.empty((0, N_PARAMS), np.float32)
    spectra = np.concatenate(spectra_chunks) if spectra_chunks else np.empty((0, n_k), np.float32)
    del params_chunks, spectra_chunks
    gc.collect()

    assert spectra.shape[1] == n_k, f"Spectrum cols {spectra.shape[1]} != k-modes {n_k}"

    n_valid = len(params)
    print(f"  Valid: {n_valid:,}, removed: {n_removed:,} "
          f"({100 * n_removed / max(total, 1):.2f}%)")

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
    print(f"Loaded {n_k} k-modes from {K_FILE}\n")

    for quantity, split, filename in FILES:
        if not os.path.exists(filename):
            print(f"SKIP {filename} (not found -- single-network run is OK)")
            continue
        out_params = os.path.join(OUTPUT_DIR,
                                  f"camb_{quantity}_params_{split}.npy")
        out_features = os.path.join(OUTPUT_DIR,
                                    f"camb_{quantity}_logpower_{split}.npy")
        if os.path.exists(out_params) and os.path.exists(out_features):
            print(f"SKIP {quantity} {split} (output already exists)")
            continue
        print(f"{quantity} {split}:")
        convert(filename, quantity, split, n_k, k_modes)
        print()

    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()