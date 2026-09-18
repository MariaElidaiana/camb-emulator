#!/usr/bin/env python
"""
Train CosmoPower NN emulator on v2 CAMB data (z=0, 6 parameters).

Differences from train_emulator.py:
- 6 parameters (no z): h0, omega_m, omega_b, n_s, log1e10As, mnu
- DATA_DIR = ./training_data_v2
- SPECTRA choices: linear_v2, linear_nonu_v2
- Model output: camb_{spectra}_emulator.pkl in the current dir

Usage:
    python train_emulator_v2.py --spectra linear_v2 [--nsamples N]
"""
import argparse
import os
import time
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from cosmopower import cosmopower_NN

parser = argparse.ArgumentParser(description="Train CosmoPower emulator (v2)")
parser.add_argument("--nsamples", type=int, default=0,
                    help="Number of training samples (0 = use all)")
parser.add_argument("--spectra", type=str, default="linear_v2",
                    choices=["linear_v2", "linear_nonu_v2",
                             "linear_v2c", "linear_nonu_v2c",
                             "linear_v3c", "linear_nonu_v3c",
                             "linear_nl_v2", "linear_nonu_nl_v2",
                             "linear_nl_v3c", "linear_nonu_nl_v3c"],
                    help="Spectra type (default: linear_v2)")
parser.add_argument("--data-dir", type=str, default=None,
                    help="Override DATA_DIR (default: ./training_data_v2 "
                         "for *_v2, ./training_data_v2c for *_v2c, "
                         "./training_data_v3c for *_v3c, "
                         "./training_data_nl_v2 for *_nl_v2, "
                         "./training_data_nl_v3c for *_nl_v3c)")
parser.add_argument("--max-epochs", type=str, default=None,
                    help="Override max-epochs schedule as 'e1,e2,e3'. "
                         "Useful for short debug runs that can't fit the "
                         "default 400/800/1200 schedule in the wall-time "
                         "cap (e.g. '--max-epochs 50,100,150').")
args = parser.parse_args()

if args.data_dir is not None:
    DATA_DIR = args.data_dir
elif args.spectra.endswith("_nl_v3c"):
    DATA_DIR = "./training_data_nl_v3c"
elif args.spectra.endswith("_nl_v2"):
    DATA_DIR = "./training_data_nl_v2"
elif args.spectra.endswith("_v3c"):
    DATA_DIR = "./training_data_v3c"
elif args.spectra.endswith("_v2c"):
    DATA_DIR = "./training_data_v2c"
else:
    DATA_DIR = "./training_data_v2"
SPECTRA_TYPE = args.spectra
MODEL_NAME = f"camb_{SPECTRA_TYPE}_emulator"

BASE_PARAMETERS = ['h0', 'omega_m', 'omega_b', 'n_s', 'log1e10As', 'mnu']
# "_nl_" marks z-aware nonlinear emulators (both _nl_v2 and _nl_v3c).
NL_VARIANT = "_nl_" in args.spectra
if NL_VARIANT:
    MODEL_PARAMETERS = BASE_PARAMETERS + ['z']
else:
    MODEL_PARAMETERS = BASE_PARAMETERS

HIDDEN_LAYERS = [512, 512, 512, 512]

LEARNING_RATES = [1e-3, 3e-4, 1e-4]
BATCH_SIZES_LARGE  = [200000, 500000, 1000000]
BATCH_SIZES_MEDIUM = [5000, 10000, 50000]
BATCH_SIZES_SMALL  = [1000, 5000, 10000]
MAX_EPOCHS = [400, 800, 1200]
PATIENCE = [30, 30, 30]

LARGE_DATASET_THRESHOLD  = 5_000_000
MEDIUM_DATASET_THRESHOLD = 500_000

gpus = tf.config.list_physical_devices('GPU')
n_gpus = len(gpus)
print(f"GPUs available: {n_gpus}")
if gpus:
    for gpu in gpus:
        print(f"  {gpu.name}")
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")

if n_gpus >= 1:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    print(f"Using single GPU: {gpus[0].name}")
GPU_SCALE = 1

np.random.seed(42)
tf.random.set_seed(42)

print(f"\nLoading {SPECTRA_TYPE} training data from {DATA_DIR}...")
t0 = time.time()

params_npy   = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_params_train.npy")
features_npy = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_logpower_train.npy")
modes_npy    = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_modes.npy")

if not (os.path.exists(params_npy) and os.path.exists(features_npy)):
    raise FileNotFoundError(
        f"No training data in {DATA_DIR}. Run clean_and_split_data_v2.py first."
    )

params_arr = np.load(params_npy, mmap_mode='r')
training_features_arr = np.load(features_npy, mmap_mode='r')
modes = np.load(modes_npy)
training_parameters = {name: params_arr[:, i] for i, name in enumerate(MODEL_PARAMETERS)}

t_load = time.time() - t0
n_total = len(training_features_arr)
print(f"Loaded {n_total:,} samples, {len(modes)} k-modes in {t_load:.1f}s")

if args.nsamples > 0 and args.nsamples < n_total:
    print(f"Subsampling to {args.nsamples:,} samples...")
    idx = np.random.choice(n_total, args.nsamples, replace=False)
    idx.sort()
    training_features_arr = np.array(training_features_arr[idx])
    training_parameters = {name: np.array(training_parameters[name])[idx]
                           for name in MODEL_PARAMETERS}
    n_total = args.nsamples

for name in MODEL_PARAMETERS:
    arr = training_parameters[name]
    print(f"  {name}: [{arr.min():.4f}, {arr.max():.4f}]")

if n_total >= LARGE_DATASET_THRESHOLD:
    batch_sizes = [bs * GPU_SCALE for bs in BATCH_SIZES_LARGE]
    print(f"\nUsing LARGE batch schedule")
elif n_total >= MEDIUM_DATASET_THRESHOLD:
    batch_sizes = [bs * GPU_SCALE for bs in BATCH_SIZES_MEDIUM]
    print(f"\nUsing MEDIUM batch schedule")
else:
    batch_sizes = [bs * GPU_SCALE for bs in BATCH_SIZES_SMALL]
    print(f"\nUsing SMALL batch schedule")

n_train = int(n_total * 0.8)
steps_per_epoch = [n_train // bs for bs in batch_sizes]
print(f"  Steps per epoch by phase: {steps_per_epoch}")

print(f"\nInitializing CosmoPower NN...")
print(f"  Parameters: {MODEL_PARAMETERS}")
print(f"  Hidden layers: {HIDDEN_LAYERS}")
print(f"  Training samples: {n_total:,}")

if args.max_epochs is not None:
    max_epochs_override = [int(s) for s in args.max_epochs.split(",")]
    if len(max_epochs_override) != len(MAX_EPOCHS):
        raise SystemExit(
            f"--max-epochs must have {len(MAX_EPOCHS)} entries "
            f"(got {len(max_epochs_override)} from '{args.max_epochs}')"
        )
    MAX_EPOCHS = max_epochs_override
    print(f"  Override MAX_EPOCHS -> {MAX_EPOCHS} (debug schedule)")

checkpoint_file = MODEL_NAME

def build_model():
    if os.path.exists(checkpoint_file + '.pkl'):
        print(f"  Found existing checkpoint: {checkpoint_file}.pkl -- resuming")
        return cosmopower_NN(
            parameters=MODEL_PARAMETERS, modes=modes, n_hidden=HIDDEN_LAYERS,
            verbose=True, restore=True, restore_filename=checkpoint_file,
        )
    return cosmopower_NN(
        parameters=MODEL_PARAMETERS, modes=modes, n_hidden=HIDDEN_LAYERS,
        verbose=True,
    )

cp_nn = build_model()

print(f"\nStarting training...")
t_train = time.time()
cp_nn.train(
    training_parameters=training_parameters,
    training_features=training_features_arr,
    filename_saved_model=MODEL_NAME,
    validation_split=0.2,
    learning_rates=LEARNING_RATES,
    batch_sizes=batch_sizes,
    gradient_accumulation_steps=[1, 1, 1],
    patience_values=PATIENCE,
    max_epochs=MAX_EPOCHS,
)
print(f"\nTraining complete! ({time.time() - t_train:.1f}s)")
print(f"Model saved as: {MODEL_NAME}")

print(f"\n{'='*60}\nEvaluating on test set...")
test_params_npy   = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_params_test.npy")
test_features_npy = os.path.join(DATA_DIR, f"camb_{SPECTRA_TYPE}_logpower_test.npy")

if not (os.path.exists(test_params_npy) and os.path.exists(test_features_npy)):
    print("No test data found, skipping evaluation.")
else:
    test_params_arr = np.load(test_params_npy)
    test_features_true = np.load(test_features_npy)
    test_parameters = {name: test_params_arr[:, i] for i, name in enumerate(MODEL_PARAMETERS)}

    n_test = len(test_features_true)
    print(f"  Test samples: {n_test:,}")

    EVAL_BATCH = 50000
    predictions = []
    for i in range(0, n_test, EVAL_BATCH):
        batch_params = {name: test_parameters[name][i:i+EVAL_BATCH]
                        for name in MODEL_PARAMETERS}
        predictions.append(cp_nn.predictions_np(batch_params))
    test_features_pred = np.concatenate(predictions, axis=0)

    residuals = test_features_pred - test_features_true
    rmse_log = np.sqrt(np.mean(residuals**2))
    frac_error = np.abs(np.power(10.0, np.clip(residuals, -30, 30)) - 1.0)
    median_frac = np.median(frac_error)
    pct95_frac = np.percentile(frac_error, 95)
    pct99_frac = np.percentile(frac_error, 99)

    within_01 = np.mean(frac_error < 0.01) * 100
    within_05 = np.mean(frac_error < 0.05) * 100
    within_10 = np.mean(frac_error < 0.10) * 100

    print(f"\n  Results on test set ({n_test:,} samples, {len(modes)} k-modes):")
    print(f"  RMSE [log10 P(k)]:   {rmse_log:.6f}")
    print(f"  |dP/P| median:       {median_frac:.4%}")
    print(f"  |dP/P| 95th pct:     {pct95_frac:.4%}")
    print(f"  |dP/P| 99th pct:     {pct99_frac:.4%}")
    print(f"  < 1%%:  {within_01:.1f}%   < 5%%:  {within_05:.1f}%   < 10%%:  {within_10:.1f}%")
