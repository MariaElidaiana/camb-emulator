"""
Generate report plots for the v3c NONLINEAR emulators (P_nl(k,z)).

Unlike make_report_plots.py (linear v2c, 6 params, z=0), the nonlinear
emulators take z as a seventh input feature and emulate P_nl(k,z)
directly from CAMB halofit (takahashi). The test set is ~1.1M rows
(100k cosmologies x 11 z-slices), so this script computes accuracy
metrics on the FULL test set while generating the figures from a
subsample to keep memory bounded.

Outputs (into report/plots/):
  err_cdf.pdf      — |dP/P| CDF for both emulators (subsample)
  err_vs_k.pdf     — median and 95th pct |dP/P| vs k (subsample)
  examples.pdf     — 4 (z, Omega_m) points, truth vs predicted, residuals

Also prints the full-test accuracy table (RMSE, median/95th/99th |dP/P|,
fraction < 1%/5%) for copy-paste into report/emulator_nl_report.tex.

Usage (from the repo root, cosmopower env):
  $JESTEVES_COSMOPOWER_PY scripts/make_report_plots_nl.py
"""
from __future__ import annotations

import gc
import os

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from cosmopower import cosmopower_NN

DATA_DIR = "./training_data_nl_v3c"
PLOT_DIR = "./report/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Order must match clean_and_split_data_nl_v3c.py / train_emulator_v2.py:
# the 7th column is z.
PARAMS = ["h0", "omega_m", "omega_b", "n_s", "log1e10As", "mnu", "z"]

EMULATORS = [
    ("linear_nl_v3c",      "camb_linear_nl_v3c_emulator",
     r"$P^{\rm mm}_{\rm nl}(k,z)$"),
    ("linear_nonu_nl_v3c", "camb_linear_nonu_nl_v3c_emulator",
     r"$P^{\rm cb}_{\rm nl}(k,z)$"),
]

COLORS = {"linear_nl_v3c": "C0", "linear_nonu_nl_v3c": "C3"}

# Subsample size for the figures (full test set is ~1.1M rows).
NPLOT_SUBSAMPLE = 200_000
_LOG10_F32 = np.float32(2.302585092994046)


def predict(cp_nn, params_arr, batch=20000):
    out = []
    for i in range(0, len(params_arr), batch):
        p = {n: params_arr[i:i + batch, j] for j, n in enumerate(PARAMS)}
        out.append(cp_nn.predictions_np(p))
    return np.concatenate(out, axis=0).astype(np.float32)


def frac_error(log_pred, log_true):
    r = log_pred - log_true
    return np.abs(np.exp(np.clip(r, -30.0, 30.0) * _LOG10_F32) - 1.0)


def pick_examples(params, logp_true, logp_pred):
    """Return [(title, row_index, z)]: 2 z-values x low/high Omega_m."""
    z = params[:, 6].astype(np.float64)
    om = params[:, 1].astype(np.float64)
    picks = []
    for z_target in (0.0, 1.0):
        dz = np.abs(z - z_target)
        idx_z = np.where(dz < 1e-3)[0]
        if len(idx_z) == 0:
            idx_z = np.array([int(np.argmin(dz))])
            z_target = float(z[idx_z[0]])
        om_z = om[idx_z]
        ilo = idx_z[np.argmin(om_z)]
        ihi = idx_z[np.argmax(om_z)]
        picks.append((f"$z={z_target:.1f}$, low $\\Omega_m$", ilo, z_target))
        picks.append((f"$z={z_target:.1f}$, high $\\Omega_m$", ihi, z_target))
    return picks


def evaluate(spectra, ckpt, label):
    print(f"Loading {ckpt} ...")
    cp_nn = cosmopower_NN(restore=True, restore_filename=ckpt)

    params = np.load(os.path.join(DATA_DIR, f"camb_{spectra}_params_test.npy"))
    logp_true = np.load(os.path.join(DATA_DIR, f"camb_{spectra}_logpower_test.npy"))
    n_test = len(params)
    print(f"  n_test={n_test:,}")

    logp_pred = predict(cp_nn, params)

    # ---- full-test metrics (exact) ----
    r = logp_pred - logp_true
    rmse = float(np.sqrt(np.mean(r.astype(np.float64) ** 2)))
    del r

    fe = frac_error(logp_pred, logp_true)
    median = 100 * float(np.percentile(fe, 50))
    p95 = 100 * float(np.percentile(fe, 95))
    p99 = 100 * float(np.percentile(fe, 99))
    frac1 = 100 * float(np.mean(fe < 0.01))
    frac5 = 100 * float(np.mean(fe < 0.05))
    del fe
    gc.collect()

    print(f"  RMSE[log10 P]={rmse:.4f}  median|dP/P|={median:.4f}%  "
          f"95th={p95:.4f}%  99th={p99:.4f}%  <1%={frac1:.2f}%  <5%={frac5:.2f}%")

    # ---- subsample for figures (bounded memory) ----
    if n_test > NPLOT_SUBSAMPLE:
        rng = np.random.default_rng(20260507)
        idx = rng.choice(n_test, NPLOT_SUBSAMPLE, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n_test)
    fe_small = frac_error(logp_pred[idx], logp_true[idx])

    # ---- 4 example rows (store small truth/pred slices) ----
    examples = pick_examples(params, logp_true, logp_pred)
    example_rows = [e[1] for e in examples]
    ex = {
        "titles": [e[0] for e in examples],
        "zvals": [e[2] for e in examples],
        "logp_true": logp_true[example_rows],
        "logp_pred": logp_pred[example_rows],
    }

    del params, logp_true, logp_pred
    gc.collect()

    return {
        "label": label,
        "rmse": rmse,
        "median": median, "p95": p95, "p99": p99,
        "frac1": frac1, "frac5": frac5,
        "fe_small": fe_small,
        "examples": ex,
        "n_test": n_test,
    }


def main():
    k = np.load(os.path.join(DATA_DIR, "camb_linear_nl_v3c_modes.npy"))

    results = {}
    for spectra, ckpt, label in EMULATORS:
        results[spectra] = evaluate(spectra, ckpt, label)
        print()

    # ---- metrics table (print) ----
    print("=" * 70)
    print("Accuracy (full test set). Copy into emulator_nl_report.tex:")
    print("=" * 70)
    for spectra, r in results.items():
        print(f"{spectra:20s}  n_test={r['n_test']:,}")
        print(f"  RMSE={r['rmse']:.4f}  median={r['median']:.4f}%  "
              f"95th={r['p95']:.4f}%  99th={r['p99']:.4f}%  "
              f"<1%={r['frac1']:.2f}%  <5%={r['frac5']:.2f}%")

    # ---- 1) CDF of |dP/P| ----
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for spectra, r in results.items():
        flat = r["fe_small"].ravel()
        xs = np.sort(flat)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        idx = np.linspace(0, len(xs) - 1, 4000).astype(int)
        ax.semilogx(xs[idx] * 100, ys[idx],
                    color=COLORS[spectra], lw=1.8, label=r["label"])
    for t, ls in [(0.01, ":"), (0.05, "--")]:
        ax.axvline(t * 100, color="gray", ls=ls, lw=0.8)
    ax.set_xlabel(r"$|P_{\rm pred}/P_{\rm true} - 1|$ [%]")
    ax.set_ylabel(r"CDF (per-$k$, per-cosmology)")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(0, 1.01)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "err_cdf.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- 2) |dP/P| vs k ----
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for spectra, r in results.items():
        med = np.median(r["fe_small"], axis=0) * 100
        p95 = np.percentile(r["fe_small"], 95, axis=0) * 100
        ax.loglog(k, med, color=COLORS[spectra], lw=1.8,
                  label=f"{r['label']} median")
        ax.loglog(k, p95, color=COLORS[spectra], lw=1.2, ls="--",
                  label=f"{r['label']} 95th")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel(r"$|P_{\rm pred}/P_{\rm true} - 1|$ [%]")
    ax.set_xlim(k.min(), k.max())
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "err_vs_k.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---- 3) example spectra (z x Omega_m) ----
    r0 = results["linear_nl_v3c"]
    n_ex = len(r0["examples"]["titles"])

    fig, axes = plt.subplots(2, n_ex, figsize=(2.7 * n_ex, 4.6), sharex="col",
                             gridspec_kw={"height_ratios": [2.4, 1.0]})
    for col in range(n_ex):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        for spectra, r in results.items():
            pk_true = 10 ** r["examples"]["logp_true"][col]
            pk_pred = 10 ** r["examples"]["logp_pred"][col]
            c = COLORS[spectra]
            ax_top.loglog(k, pk_true, color=c, lw=1.4, ls="-",
                          label=f"{r['label']} truth" if col == 0 else None)
            ax_top.loglog(k, pk_pred, color=c, lw=1.0, ls="--",
                          label=f"{r['label']} emu" if col == 0 else None)
            ax_bot.semilogx(k, (pk_pred / pk_true - 1.0) * 100,
                            color=c, lw=1.2)
        ax_top.set_title(r0["examples"]["titles"][col], fontsize=9)
        ax_top.grid(True, which="both", alpha=0.25)
        ax_bot.grid(True, which="both", alpha=0.25)
        ax_bot.axhline(0, color="gray", lw=0.6)
        ax_bot.set_xlim(k.min(), k.max())
        ax_bot.set_ylim(-2.5, 2.5)
        ax_bot.set_xlabel(r"$k$ [$h$/Mpc]")
        if col == 0:
            ax_top.set_ylabel(r"$P_{\rm nl}(k,z)$ [$({\rm Mpc}/h)^3$]")
            ax_bot.set_ylabel(r"$P_{\rm pred}/P_{\rm true} - 1$ [%]")
    axes[0, 0].legend(loc="lower left", fontsize=7, frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "examples.pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote plots to {PLOT_DIR}/")


if __name__ == "__main__":
    main()