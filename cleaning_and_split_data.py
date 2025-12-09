import numpy as np
from sklearn.model_selection import train_test_split

# redefine 
k_modes = np.loadtxt('k_modes.txt')

# load predictions from Class: concatenations of parameters and spectra
linear_spectra_and_params = np.loadtxt('./linear_combined.dat')
boost_spectra_and_params = np.loadtxt('./boost_combined.dat')

# clean NaN's if any
rows = np.where(np.isfinite(linear_spectra_and_params).all(1))
linear_spectra_and_params = linear_spectra_and_params[rows]

rows = np.where(np.isfinite(boost_spectra_and_params).all(1))
boost_spectra_and_params = boost_spectra_and_params[rows]

# here the ordering should match the one used in `1_create_params.py`
params = ['omega_b', 'omega_cdm', 'h', 'n_s', 'ln10^{10}A_s', 'c_min', 'eta_0', 'z']
n_params = len(params)

# separate parameters from spectra, take log
linear_parameters = linear_spectra_and_params[:, :n_params]
linear_log_spectra = np.log10(linear_spectra_and_params[:, n_params:])

boost_parameters = boost_spectra_and_params[:, :n_params]
boost_log_spectra = np.log10(boost_spectra_and_params[:, n_params:])

# --- Split into train/test (80/20) ---
linear_params_train, linear_params_test, linear_spectra_train, linear_spectra_test = train_test_split(
    linear_parameters, linear_log_spectra, test_size=0.2, random_state=42
)

boost_params_train, boost_params_test, boost_spectra_train, boost_spectra_test = train_test_split(
    boost_parameters, boost_log_spectra, test_size=0.2, random_state=42
)

# --- Prepare dictionaries for saving ---
def make_dicts(parameters, spectra):
    parameters_dict = {params[i]: parameters[:, i] for i in range(len(params))}
    spectra_dict = {'modes': k_modes, 'features': spectra}
    return parameters_dict, spectra_dict

linear_parameters_dict_train, linear_log_spectra_dict_train = make_dicts(linear_params_train, linear_spectra_train)
linear_parameters_dict_test, linear_log_spectra_dict_test = make_dicts(linear_params_test, linear_spectra_test)

boost_parameters_dict_train, boost_log_spectra_dict_train = make_dicts(boost_params_train, boost_spectra_train)
boost_parameters_dict_test, boost_log_spectra_dict_test = make_dicts(boost_params_test, boost_spectra_test)

# --- Save train and test sets ---
np.savez('class_linear_params_train.npz', **linear_parameters_dict_train)
np.savez('class_linear_logpower_train.npz', **linear_log_spectra_dict_train)
np.savez('class_linear_params_test.npz', **linear_parameters_dict_test)
np.savez('class_linear_logpower_test.npz', **linear_log_spectra_dict_test)

np.savez('class_boost_params_train.npz', **boost_parameters_dict_train)
np.savez('class_boost_logpower_train.npz', **boost_log_spectra_dict_train)
np.savez('class_boost_params_test.npz', **boost_parameters_dict_test)
np.savez('class_boost_logpower_test.npz', **boost_log_spectra_dict_test)

