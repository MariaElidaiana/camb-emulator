import os
import warnings
#To suppress the noisy CUDA messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = info, 2 = warning, 3 = error
warnings.filterwarnings('ignore')         # Suppress Python warnings
import tensorflow as tf
import numpy as np
from cosmopower import cosmopower_NN

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if gpus else '/CPU:0'
print(f'Using device: {device}')

# Enable mixed precision for speedup
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled:", mixed_precision.global_policy())

# Seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(2)

# Load training parameters
training_parameters = np.load('./class_linear_params_train.npz')
print(training_parameters.files)

# Load training features
training_features = np.load('./class_linear_logpower_train.npz')
print(training_features.files)
training_log_spectra = training_features['features']
print('Number of wavenumbers:', len(training_features['modes']))
print('Training shape:', training_log_spectra.shape)

# Model parameters
model_parameters = ['z', 'h', 'eta_0', 'c_min', 'omega_b', 'n_s', 'ln10^{10}A_s', 'omega_cdm']

# k-modes
k_range = training_features['modes']

# Instantiate NN
cp_nn = cosmopower_NN(
    parameters=model_parameters,
    modes=k_range,
    n_hidden=[512, 512, 512, 512],
    verbose=True
)

# TRAINING with progressive batch sizes, adaptive LR, mixed precision
with tf.device(device):
    cp_nn.train(
        training_parameters=training_parameters,
        training_features=training_log_spectra,
        filename_saved_model='Pk_cp_NN_optmized_Maria17Oct2025_z',

        validation_split=0.2,  # 20% for validation

        # specify the LRs in decreasing order
        learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],

        # Progressive batch sizes: increase as LR decreases for better GPU utilization
        batch_sizes=[1000, 10000, 20000, 40000, 50000],

        # Gradient accumulation steps: keep 1
        gradient_accumulation_steps=[1, 1, 1, 1, 1],

        # Early stopping patience per LR phase
        patience_values=[20, 20, 20, 20, 20],

        # Max epochs per LR phase
        max_epochs=[100, 200, 300, 500, 1000],

    )
