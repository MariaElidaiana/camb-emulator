# camb_class_halofit_compare.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# --------- Imports ---------
try:
    import camb
    from camb import model
except Exception as e:
    raise ImportError("CAMB is required. Install via `pip install camb`.") from e

try:
    from classy import Class
    CLASS_AVAILABLE = True
except Exception:
    CLASS_AVAILABLE = False

# --------- Planck 2018 baseline ΛCDM ---------
A_s = 2.100549e-9
h = 0.6736
H0 = 100.0 * h
ombh2 = 0.02237
omch2 = 0.1200
n_s = 0.9649
tau = 0.0544

# neutrinos
m_nu = 0.06
N_ncdm = 1
N_ur = 2.046

# redshifts and k-range
redshifts = [0.0, 0.8]
camb_minkh = 1e-4
camb_maxkh = 1.0
n_kpoints = 400

# --------- 1) CAMB (Halofit) ---------
print("Running CAMB (Halofit)...")
start = time.time()

pars = camb.set_params(
    H0=H0,
    ombh2=ombh2,
    omch2=omch2,
    ns=n_s,
    As=A_s,
    tau=tau,
    pivot_scalar=0.05,
    mnu=m_nu,
    num_massive_neutrinos=N_ncdm,
    nnu=3.046
)

pars.set_matter_power(redshifts=redshifts, kmax=2.0)

# Linear spectrum
pars.NonLinear = model.NonLinear_none
results_linear = camb.get_results(pars)
kh, z_camb, pk_lin = results_linear.get_matter_power_spectrum(
    minkh=camb_minkh, maxkh=camb_maxkh, npoints=n_kpoints
)

# Nonlinear spectrum using Halofit
pars.NonLinear = model.NonLinear_both
results_nl = camb.get_results(pars)
kh_nl, z_camb_nl, pk_nl = results_nl.get_matter_power_spectrum(
    minkh=camb_minkh, maxkh=camb_maxkh, npoints=n_kpoints
)

end = time.time()
print(f"CAMB finished in {end - start:.3f} seconds.")
print("CAMB sigma8 (non-linear):", results_nl.get_sigma8())

# Convert CAMB units to CLASS units
k_camb_phys = kh * h
pk_lin_camb_phys = pk_lin / h**3
pk_nl_camb_phys = pk_nl / h**3

# --------- 2) CLASS (Halofit) ---------
if not CLASS_AVAILABLE:
    print("CLASS not installed. Skipping CLASS calculation.")
else:
    print("\nRunning CLASS (Halofit)...")
    cosmo = Class()
    class_params = {
        'output': 'mPk',
        'h': h,
        'omega_b': ombh2,
        'omega_cdm': omch2,
        'A_s': A_s,
        'n_s': n_s,
        'tau_reio': tau,
        'k_pivot': 0.05,
        'm_ncdm': str(m_nu),
        'N_ncdm': str(N_ncdm),
        'N_ur': str(N_ur),
        'non linear': 'halofit',
        'P_k_max_1/Mpc': 5.0,
        'z_pk': ','.join([f"{z:.6f}" for z in redshifts])
    }

    start = time.time()

    cosmo_linear = Class()
    class_params_linear = class_params.copy()
    class_params_linear['non linear'] = 'none'  # linear only
    cosmo_linear.set(class_params_linear)
    cosmo_linear.compute()

    cosmo_nl = Class()
    class_params_nl = class_params.copy()
    class_params_nl['non linear'] = 'halofit'
    cosmo_nl.set(class_params_nl)
    cosmo_nl.compute()


    # get CLASS P(k) on CAMB k-grid
    pk_class = []
    pk_nl_class = []
    for zi in redshifts:
        # Linear P(k)
        pkz = np.array([cosmo_linear.pk(k, zi) for k in k_camb_phys])
        pk_class.append(pkz)

        # Non-Linear P(k)
        pkz_nl = np.array([cosmo_nl.pk(k, zi) for k in k_camb_phys])
        pk_nl_class.append(pkz_nl)

    pk_class = np.array(pk_class)
    pk_nl_class = np.array(pk_nl_class)

    end = time.time()
    print(f"CLASS finished in {end - start:.3f} seconds.")

    # --------- 3) PLOTS ---------
    #plt.figure(figsize=(8,6))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, zi in enumerate(redshifts):
        ax = axes[i]
        ax.loglog(k_camb_phys, pk_lin_camb_phys[i], color='k', ls='-', label=f'CAMB Linear z={zi}')
        ax.loglog(k_camb_phys, pk_class[i],       color='r', ls='--', label=f'CLASS Linear z={zi}')
        ax.loglog(k_camb_phys, pk_nl_camb_phys[i], color='k', ls=':', label=f'CAMB Non-Linear z={zi}')
        ax.loglog(k_camb_phys, pk_nl_class[i],       color='r', ls='-.', label=f'CLASS Non-Linear z={zi}')
        ax.set_ylim(1e1, 1e5)
        ax.set_xlabel(r'$k\ [\mathrm{Mpc}^{-1}]$')
        ax.set_ylabel(r'$P(k)\ [\mathrm{Mpc}^3]$')
        ax.set_title(f'z = {zi}')
        ax.legend()
    plt.suptitle('CAMB vs CLASS (Halofit) — Planck2018', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.tight_layout()
    plt.savefig('Pk_comp.png', dpi=300)

    # Relative difference
    plt.figure(figsize=(8,5))
    for i, zi in enumerate(redshifts):
        rel = (pk_nl_class[i] - pk_nl_camb_phys[i]) / pk_nl_camb_phys[i]
        plt.semilogx(k_camb_phys, rel, label=f'z={zi}')
    plt.axhline(0, color='k', lw=0.8)
    plt.xlabel(r'$k\ [\mathrm{Mpc}^{-1}]$')
    plt.ylabel('Relative difference (CLASS - CAMB) / CAMB')
    plt.title('Relative difference (Non-Linear): CLASS(Halofit) vs CAMB(Halofit)')
    plt.legend()
    plt.grid(axis='x', which='both', alpha=0.2)
    plt.tight_layout()
    plt.savefig('Pk_comp_reladiff_nl.png', dpi=300)
    
    # Clean up
    cosmo_linear.struct_cleanup()
    cosmo_linear.empty()
    cosmo_nl.struct_cleanup()
    cosmo_nl.empty()
