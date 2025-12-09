import numpy as np
from classy import Class
from multiprocessing import Pool, cpu_count
from mpi4py import MPI
import os, time

# -------------------------
# MPI Initialization
# -------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------------
# Precompute k-modes once
# -------------------------
if rank == 0:
    krange1 = np.logspace(np.log10(1e-5), np.log10(1e-4), num=20, endpoint=False)
    krange2 = np.logspace(np.log10(1e-4), np.log10(1e-3), num=40, endpoint=False)
    krange3 = np.logspace(np.log10(1e-3), np.log10(1e-2), num=60, endpoint=False)
    krange4 = np.logspace(np.log10(1e-2), np.log10(1e-1), num=80, endpoint=False)
    krange5 = np.logspace(np.log10(1e-1), np.log10(1), num=100, endpoint=False)
    krange6 = np.logspace(np.log10(1), np.log10(10), num=120, endpoint=False)
    k = np.concatenate((krange1, krange2, krange3, krange4, krange5, krange6))
    np.savetxt('k_modes.txt', k)
else:
    k = None

# Broadcast k to all ranks
k = comm.bcast(k, root=0)

# -------------------------
# Load parameter file fully (only once on rank 0, then broadcast)
# -------------------------
if rank == 0:
    params_file = np.load('tutorial8p_LHS_parameter_file.npz')
    params_lhs = {key: params_file[key] for key in params_file.files}
else:
    params_lhs = None

params_lhs = comm.bcast(params_lhs, root=0)
total = len(params_lhs['omega_b'])

# -------------------------
# Divide work among MPI ranks
# -------------------------
chunk_size = total // size
remainder = total % size
start_idx = rank * chunk_size + min(rank, remainder)
end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)
indices = list(range(start_idx, end_idx))

# -------------------------
# Worker function
# -------------------------
def spectra_generation(i):
    cosmo = Class()
    params = {
        'output': 'tCl mPk',
        'non linear': 'hmcode',
        'z_max_pk': 5,
        'P_k_max_1/Mpc': 10.,
        'nonlinear_min_k_max': 100.,
        'N_ncdm': 0,
        'Neff': 3.046,
        'omega_b': params_lhs['omega_b'][i],
        'omega_cdm': params_lhs['omega_cdm'][i],
        'h': params_lhs['h'][i],
        'n_s': params_lhs['n_s'][i],
        'ln10^{10}A_s': params_lhs['ln10^{10}A_s'][i]
    }

    try:
        cosmo.set(params)
        cosmo.compute()
        z = params_lhs['z'][i]

        Plin = np.array([cosmo.pk_lin(ki, z) for ki in k])
        Pnonlin = np.array([cosmo.pk(ki, z) for ki in k])

        lin_array = np.hstack(([params_lhs[key][i] for key in params_lhs], Plin))
        boost_array = np.hstack(([params_lhs[key][i] for key in params_lhs], Pnonlin / Plin))

        # Write chunk-specific files to avoid race conditions
        with open(f'linear_rank{rank}.dat', 'ab') as f_lin, \
             open(f'boost_rank{rank}.dat', 'ab') as f_boost:
            np.savetxt(f_lin, [lin_array])
            np.savetxt(f_boost, [boost_array])

    except Exception as e:
        print(f"[Rank {rank}] Warning: Parameter set {i} failed: {e}")
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    comm.Barrier()
    start = time.time()

    print(f"[Rank {rank}] Handling indices {start_idx}:{end_idx} on {cpu_count()} local cores.")

    with Pool(processes=cpu_count()) as pool:
        pool.map(spectra_generation, indices)

    comm.Barrier()
    end = time.time()

    if rank == 0:
        print(f"Finished {total} parameter sets using {size} MPI ranks in {(end - start)/3600:.2f} hours.")

