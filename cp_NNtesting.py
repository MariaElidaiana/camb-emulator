import os
#To suppress the noisy CUDA messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = info, 2 = warning, 3 = error
import tensorflow as tf
import numpy as np
from cosmopower import cosmopower_NN

# Checking that we are using a GPU
gpus = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if gpus else '/CPU:0'
print(f'Using device: {device}')

# setting the seed for reproducibility
np.random.seed(1)
tf.random.set_seed(2)

# training parameters
training_parameters = np.load('./class_linear_params_train.npz')
print(training_parameters.files)

# training features (= log-spectra, in this case)
training_features = np.load('class_linear_logpower_train.npz')
print(training_features.files)
# Info: modes
#print(training_features['modes'])
print('number of multipoles: ', len(training_features['modes']))
# Info: features
training_log_spectra = training_features['features']
print('(number of training samples, number of k-modes): ', training_log_spectra.shape)

# MODES
k_range = training_features['modes']

# testing parameters
testing_params = np.load('./class_linear_params_test.npz')
# testing features
testing_spectra = 10.**(np.load('./class_linear_logpower_test.npz')['features'])

cp_nn = cosmopower_NN(restore=True,
                      restore_filename='Pk_cp_NN_optmized_Maria17Oct2025_z', # With fixed training mit z
                      )

print('emulator parameters: ', cp_nn.parameters)
print('sampled k-modes: ', cp_nn.modes)
print('number of k-modes: ', cp_nn.n_modes)
print('hidden layers: ', cp_nn.n_hidden)

# Prediction
predicted_testing_spectra = cp_nn.ten_to_predictions_np(testing_params)

import matplotlib.pyplot as plt
from matplotlib import gridspec

# number of random samples
n_samples = 9
rng = np.random.default_rng(seed=42)  # reproducible random indices
random_indices = rng.choice(len(testing_spectra), size=n_samples, replace=False)

# figure setup: 3x3 grid of samples, each with 2 stacked subplots (main + residual)
fig = plt.figure(figsize=(20, 18))
outer_grid = gridspec.GridSpec(3, 3, wspace=0.3, hspace=0.4)

for i, idx in enumerate(random_indices):
    # extract true and predicted spectra
    pred = predicted_testing_spectra[idx]
    true = testing_spectra[idx]
    residual = (pred - true) / true

    # create inner grid: 2 rows (main + residual)
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_grid[i], height_ratios=[3, 1], hspace=0.05
    )

    # --- MAIN SPECTRA PLOT (log-log) ---
    ax_main = plt.Subplot(fig, inner_grid[0])
    ax_main.loglog(k_range, true, color='blue', label='Original')
    ax_main.loglog(k_range, pred, color='red', linestyle='--', label='NN reconstructed')
    ax_main.set_ylabel('$P_{\mathrm{LIN}}(k) [\mathrm{Mpc}^3]$', fontsize='large')
    ax_main.set_title(f'Sample {idx}', fontsize='large')
    ax_main.legend(fontsize=9, loc='best')
    ax_main.tick_params(axis='x', labelbottom=False)
    fig.add_subplot(ax_main)

    # --- RESIDUAL PLOT ---
    ax_resid = plt.Subplot(fig, inner_grid[1])
    ax_resid.semilogx(k_range, residual, color='gray', alpha=0.8)
    ax_resid.axhline(0, color='k', linewidth=0.8)
    ax_resid.set_xlabel('$k$ [Mpc$^{-1}]$', fontsize='large')
    ax_resid.set_ylabel('Frac. Residual', fontsize='medium')
    ax_resid.set_ylim(-0.0005, 0.0005)
    ax_resid.tick_params(axis='both', labelsize=9)
    fig.add_subplot(ax_resid)


plt.tight_layout()
plt.savefig('examples_reconstruction_Pk_opt_random9_with_residual_panels.pdf')
