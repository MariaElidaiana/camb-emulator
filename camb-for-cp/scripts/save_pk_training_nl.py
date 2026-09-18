"""
CosmoSIS module to save CAMB nonlinear spectra at z=0..1 for emulator
training (v2-nl).

Unlike save_pk_training_v2.py (which keeps only z=0 because the linear
spectrum is separable as D(z)^2 * P(k,0)), this module keeps EVERY z-slice
as a separate row -- the nonlinear spectrum is not separable in z, so
the emulator must take z as an input feature.

Reads two datablocks per cosmology:
- matter_power_nl    (delta_tot,  includes neutrinos) -> linear_nl_v2.dat
- cdm_baryon_power_nl (delta_nonu, no neutrinos)      -> linear_nonu_nl_v2.dat

If cdm_baryon_power_nl is not present in the datablock (some CAMB /
CosmoSIS interface versions only emit delta_tot when nonlinear=halofit),
the *_nonu file is skipped and a warning is printed. In that case the
downstream single-network path (linear_nl_v2 only) is still usable.

Row format (7 leading columns, then P(k) at each z-slice):
  [h0, omega_m, omega_b, n_s, log1e10As, mnu, z, P(k1), ..., P(kN)]

Note: this matches the v1-era row layout (7 cols including z), even
though save_pk_training_v2.py uses 6 cols (z dropped). The trainer and
cleaner are told to expect 7 PARAM_NAMES accordingly.

Files produced (one per task, prefixed by SAVE_PK_PREFIX):
- k_modes_v2.txt                        (rank 0 only once)
- linear_nl_v2.dat | linear_nl_v2_rank{N}.dat
- linear_nonu_nl_v2.dat | linear_nonu_nl_v2_rank{N}.dat
"""

import os
import numpy as np
from cosmosis.datablock import names

try:
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1

print(f"[save_pk_training_nl] MPI rank={RANK} size={SIZE}", flush=True)

# Must match create_lhs_params_list_v2.py and clean_and_split_data_nl.py.
# CosmoSIS lowercases parameter names in the datablock.
PARAM_NAMES = [
    "h0",
    "omega_m",
    "omega_b",
    "n_s",
    "log1e10as",
    "mnu",
]

# CosmoSIS block names. Linear counterpart uses matter_power_lin and
# cdm_baryon_power_lin (see save_pk_training_v2.py:61). The nonlinear
# pair follows the same naming.
NL_TOT_BLOCK = "matter_power_nl"
NL_NONU_BLOCK = "cdm_baryon_power_nl"

PREFIX = os.environ.get("SAVE_PK_PREFIX", "")
if SIZE > 1:
    NL_TOT_FILE  = f"{PREFIX}linear_nl_v2_rank{RANK}.dat"
    NL_NONU_FILE = f"{PREFIX}linear_nonu_nl_v2_rank{RANK}.dat"
else:
    NL_TOT_FILE  = f"{PREFIX}linear_nl_v2.dat"
    NL_NONU_FILE = f"{PREFIX}linear_nonu_nl_v2.dat"
K_FILE = f"{PREFIX}k_modes_v2.txt"


def write_rows(filename, rows):
    with open(filename, "ab") as f:
        np.savetxt(f, rows, fmt="%.8e")


def setup(options):
    files_to_clean = [NL_TOT_FILE]
    # Only clean the nonu file when we actually expect to write it.
    # We can't know that until execute runs, but cleaning an absent
    # file is a no-op, so it's harmless to attempt.
    files_to_clean.append(NL_NONU_FILE)
    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f)
    if RANK == 0 and os.path.exists(K_FILE):
        os.remove(K_FILE)
    return {"k_saved": False, "count": 0, "nonu_present": False}


def _save_k_modes_if_first(block, config):
    if not config["k_saved"]:
        k = block[NL_TOT_BLOCK, "k_h"]
        if RANK == 0:
            np.savetxt(K_FILE, k, fmt="%.10e")
            print(f"Saved {len(k)} k-modes to {K_FILE}")
        config["k_saved"] = True


def execute(block, config):
    try:
        # delta_tot nonlinear
        pk_nl_tot = block[NL_TOT_BLOCK, "P_k"]   # (nz, nk)
        z_arr     = block[NL_TOT_BLOCK, "z"]
        # k is shared with the nonu block (same output grid in CAMB).

        _save_k_modes_if_first(block, config)

        cosmo = np.array([
            block[names.cosmological_parameters, p] for p in PARAM_NAMES
        ])

        # Build one row per z-slice for the delta_tot nonlinear block.
        rows_tot = [
            np.hstack((cosmo, np.array([zv]), pk_nl_tot[iz, :]))
            for iz, zv in enumerate(z_arr)
        ]
        rows_nonu = []
        nonu_present = False

        # Probe for the CDM+baryon nonlinear block without relying on a
        # specific datablock API method name -- some CosmoSIS builds
        # expose has_value/has_section, others raise on missing keys.
        try:
            pk_nl_nonu = block[NL_NONU_BLOCK, "P_k"]
            nonu_present = True
        except Exception as e:
            print(f"WARNING: cdm_baryon_power_nl not retrievable "
                  f"({type(e).__name__}: {e}). Skipping *_nonu file "
                  f"(linear_nonu_nl_v2 will not be produced). Single-network "
                  f"matter_power_nl path is still usable.")

        if nonu_present:
            if pk_nl_nonu.shape != pk_nl_tot.shape:
                raise RuntimeError(
                    f"cdm_baryon_power_nl shape {pk_nl_nonu.shape} != "
                    f"matter_power_nl shape {pk_nl_tot.shape}"
                )
            for iz, zv in enumerate(z_arr):
                row = np.hstack((cosmo, np.array([zv]), pk_nl_nonu[iz, :]))
                rows_nonu.append(row)

        write_rows(NL_TOT_FILE, np.vstack(rows_tot))
        if nonu_present:
            write_rows(NL_NONU_FILE, np.vstack(rows_nonu))

        config["count"] += 1
        config["nonu_present"] = nonu_present
        if config["count"] % 1000 == 0:
            print(f"Rank {RANK}: processed {config['count']} cosmologies "
                  f"({len(z_arr)} z-slices each)")

        return 0

    except Exception as e:
        print(f"Error in save_pk_training_nl.execute: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cleanup(config):
    print(f"\nRank {RANK} finished: {config['count']} cosmologies")
    print(f"Output: {NL_TOT_FILE}")
    if config.get("nonu_present", False):
        print(f"        {NL_NONU_FILE}")
    if RANK == 0:
        print(f"k-modes: {K_FILE}")
    return 0
